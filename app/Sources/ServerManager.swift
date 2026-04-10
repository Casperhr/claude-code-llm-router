import Foundation

enum ServerState: Equatable {
    case stopped
    case loading(progress: String)
    case ready
    case crashed(String)

    var label: String {
        switch self {
        case .stopped: return "Stopped"
        case .loading(let progress): return progress
        case .ready: return "Ready"
        case .crashed(let msg): return "Crashed: \(msg)"
        }
    }
}

enum ModelBackend: String {
    case mlx
    case gguf            // llama.cpp + TurboQuant
    case openrouter      // cloud via OpenRouter proxy
    case smartrouter     // hybrid: local + Sonnet + Opus
    case cloudrouter     // cloud-only: Qwen + Flash + Sonnet + Opus
}

struct LocalModel {
    let shortName: String
    let fullName: String
    let path: String
    let backend: ModelBackend
    let sizeBytes: UInt64?

    var sizeLabel: String {
        guard let s = sizeBytes else { return "" }
        let gb = Double(s) / 1e9
        return String(format: "%.0fG", gb)
    }
}

class ServerManager {
    // Base directory — defaults to ~/llm-router, override with LLM_ROUTER_DIR env var
    private static let baseDir: URL = {
        if let custom = ProcessInfo.processInfo.environment["LLM_ROUTER_DIR"] {
            return URL(fileURLWithPath: custom)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("llm-router")
    }()

    private static let home = FileManager.default.homeDirectoryForCurrentUser

    private static let modelsDir = baseDir

    private static let hfCache = home.appendingPathComponent(".cache/huggingface/hub")

    private static let uvBinary: String = {
        // Search common locations for uv binary
        let candidates = [
            home.appendingPathComponent(".local/bin/uv").path,
            "/opt/homebrew/bin/uv",
            "/usr/local/bin/uv",
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
            ?? home.appendingPathComponent(".local/bin/uv").path
    }()

    private static let serveScript = baseDir.appendingPathComponent("server/serve_mlx.py").path

    // llama.cpp binaries — searches common locations
    private static let llamaServerBinary: String = {
        let candidates = [
            home.appendingPathComponent("llama-cpp-turboquant/build/bin/llama-server").path,
            "/opt/homebrew/bin/llama-server",
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
            ?? candidates[0]
    }()

    private static let llamaServerUpstream: String = {
        let candidates = [
            home.appendingPathComponent("llama-cpp/build/bin/llama-server").path,
            "/opt/homebrew/bin/llama-server",
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
            ?? candidates[0]
    }()

    private static let ggufDir = baseDir.appendingPathComponent("gguf")

    private static let proxyScript = baseDir.appendingPathComponent("app/openrouter-proxy.py").path

    private static let smartProxyScript = baseDir.appendingPathComponent("app/smart-proxy.py").path

    private static let cloudSmartProxyScript = baseDir.appendingPathComponent("app/cloud-smart-proxy.py").path

    private static let envFile: String = {
        // Check base dir first, then parent, then home
        let candidates = [
            baseDir.appendingPathComponent(".env").path,
            baseDir.deletingLastPathComponent().appendingPathComponent(".env").path,
            home.appendingPathComponent(".env").path,
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
            ?? baseDir.appendingPathComponent(".env").path
    }()

    private static func loadOpenRouterKey() -> String {
        guard let content = try? String(contentsOfFile: envFile, encoding: .utf8) else { return "" }
        for line in content.components(separatedBy: .newlines) {
            if line.hasPrefix("OPENROUTER_API_KEY=") {
                return String(line.dropFirst("OPENROUTER_API_KEY=".count))
            }
        }
        return ""
    }

    private static let analyticsFile = "/tmp/mlx-analytics.json"
    private static let logFile = "/tmp/mlx-server.log"

    let port: Int = 5005

    /// Whether TurboQuant mode is on (serves GGUF via llama.cpp with turbo3 KV cache)
    private(set) var turboQuantEnabled: Bool = false
    private(set) var activeBackend: ModelBackend = .mlx

    private(set) var state: ServerState = .stopped
    private(set) var activeModelName: String = ""
    private(set) var mlxActiveGB: Double = 0
    private(set) var memUsedGB: Double = 0
    private(set) var memTotalGB: Double = 0
    private(set) var cpuUsage: Int = 0     // 0-100
    private(set) var gpuUsage: Int = 0     // 0-100

    var onStatusChange: (() -> Void)?

    private var healthTimer: Timer?
    private var healthFailCount: Int = 0
    private var serverProcess: Process?
    private var localModelProcess: Process?  // secondary process for Smart Router's local model
    private static let localModelPort = 5006
    private var isLoading: Bool = false

    var tooltip: String {
        let backend: String
        switch activeBackend {
        case .gguf: backend = turboQuantEnabled ? "TurboQuant" : "GGUF"
        case .mlx: backend = "MLX"
        case .openrouter: backend = "OpenRouter"
        case .smartrouter: backend = "Smart Router"
        case .cloudrouter: backend = "Cloud Router"
        }
        switch state {
        case .stopped:
            return "Local LLM — Stopped"
        case .loading(let progress):
            return "Local LLM [\(backend)] — \(progress)"
        case .ready:
            return "Local LLM [\(backend)] — \(activeModelName)\nhttp://localhost:\(port)/v1"
        case .crashed(let msg):
            return "Local LLM — Down\n\(msg)"
        }
    }

    /// Timestamp when model was last switched — requests before this are ignored
    private var modelSwitchTime: Double = 0

    /// Live tok/s — from llama.cpp /metrics for GGUF, from analytics for MLX
    private(set) var liveTokS: Double?

    /// Router stats — populated when Smart/Cloud Router is active
    private(set) var routerPctLocal: Int = 0
    private(set) var routerRequests: Int = 0
    private(set) var routerSavedPct: Double = 0
    private(set) var routerCostActual: Double = 0

    /// Avg gen tok/s from last 5 requests since current model was loaded (MLX analytics)
    var avgGenTokS: Double? {
        if activeBackend == .gguf { return liveTokS }
        guard let data = readAnalytics() else { return nil }
        let recent = data.filter { ($0["ts"] as? Double ?? 0) > modelSwitchTime }
        let speeds = recent.suffix(5).compactMap { ($0["gen_tok_s"] as? Double).flatMap { $0 > 0 ? $0 : nil } }
        guard !speeds.isEmpty else { return nil }
        return speeds.reduce(0, +) / Double(speeds.count)
    }

    init() {
        memTotalGB = Double(ProcessInfo.processInfo.physicalMemory) / 1e9
        startHealthPolling()
    }

    // MARK: - Model discovery

    func availableMLXModels() -> [LocalModel] {
        var models: [LocalModel] = []
        var seenPaths = Set<String>()

        func scanHFDir(_ baseDir: URL) {
            guard let contents = try? FileManager.default.contentsOfDirectory(atPath: baseDir.path) else { return }
            for dirName in contents.sorted() where dirName.hasPrefix("models--") {
                let snapDir = baseDir.appendingPathComponent(dirName).appendingPathComponent("snapshots")
                guard let snaps = try? FileManager.default.contentsOfDirectory(atPath: snapDir.path)
                    .filter({ !$0.hasPrefix(".") })
                    .sorted(),
                      let first = snaps.last else { continue }
                let snapPath = snapDir.appendingPathComponent(first).path
                let configPath = snapPath + "/config.json"
                guard FileManager.default.fileExists(atPath: configPath) else { continue }
                guard !seenPaths.contains(snapPath) else { continue }
                seenPaths.insert(snapPath)
                let fullName = dirName.replacingOccurrences(of: "models--", with: "").replacingOccurrences(of: "--", with: "/")
                let shortName = fullName.split(separator: "/").last.map(String.init) ?? fullName
                models.append(LocalModel(shortName: shortName, fullName: fullName, path: snapPath, backend: .mlx, sizeBytes: nil))
            }
        }

        // Direct model dirs (e.g. ~/Development/models/qwen/)
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: Self.modelsDir.path) {
            for name in contents.sorted() {
                let dir = Self.modelsDir.appendingPathComponent(name)
                let configPath = dir.appendingPathComponent("config.json").path
                guard FileManager.default.fileExists(atPath: configPath) else { continue }
                guard !seenPaths.contains(dir.path) else { continue }
                seenPaths.insert(dir.path)
                models.append(LocalModel(shortName: name, fullName: name, path: dir.path, backend: .mlx, sizeBytes: nil))
            }
        }

        scanHFDir(Self.modelsDir)
        if FileManager.default.fileExists(atPath: Self.hfCache.path) {
            scanHFDir(Self.hfCache)
        }

        return models
    }

    func availableGGUFModels() -> [LocalModel] {
        var models: [LocalModel] = []
        let ggufPath = Self.ggufDir.path

        // Scan gguf dir recursively for .gguf files
        guard let enumerator = FileManager.default.enumerator(atPath: ggufPath) else { return models }
        while let file = enumerator.nextObject() as? String {
            guard file.hasSuffix(".gguf") else { continue }
            let fullPath = ggufPath + "/" + file
            let attrs = try? FileManager.default.attributesOfItem(atPath: fullPath)
            let size = attrs?[.size] as? UInt64
            let name = URL(fileURLWithPath: file).deletingPathExtension().lastPathComponent
            models.append(LocalModel(shortName: name, fullName: file, path: fullPath, backend: .gguf, sizeBytes: size))
        }

        return models.sorted { $0.shortName < $1.shortName }
    }

    /// Smart Router: hybrid local + cloud routing
    func availableSmartRouterModels() -> [LocalModel] {
        return [
            LocalModel(shortName: "Smart Router", fullName: "smart-router", path: "smart-router", backend: .smartrouter, sizeBytes: nil),
            LocalModel(shortName: "Cloud Router", fullName: "cloud-router", path: "cloud-router", backend: .cloudrouter, sizeBytes: nil),
        ]
    }

    /// Curated list of large open-source models worth testing via OpenRouter
    func availableOpenRouterModels() -> [LocalModel] {
        return [
            LocalModel(shortName: "DeepSeek R1", fullName: "deepseek/deepseek-r1-0528", path: "deepseek/deepseek-r1-0528", backend: .openrouter, sizeBytes: nil),
            LocalModel(shortName: "Qwen3 235B A22B", fullName: "qwen/qwen3-235b-a22b", path: "qwen/qwen3-235b-a22b", backend: .openrouter, sizeBytes: nil),
            LocalModel(shortName: "Qwen3 235B Thinking", fullName: "qwen/qwen3-235b-a22b-thinking-2507", path: "qwen/qwen3-235b-a22b-thinking-2507", backend: .openrouter, sizeBytes: nil),
            LocalModel(shortName: "Llama 4 Maverick", fullName: "meta-llama/llama-4-maverick", path: "meta-llama/llama-4-maverick", backend: .openrouter, sizeBytes: nil),
            LocalModel(shortName: "Llama 4 Scout", fullName: "meta-llama/llama-4-scout", path: "meta-llama/llama-4-scout", backend: .openrouter, sizeBytes: nil),
            LocalModel(shortName: "Mistral Large", fullName: "mistralai/mistral-large-2512", path: "mistralai/mistral-large-2512", backend: .openrouter, sizeBytes: nil),
            LocalModel(shortName: "Hermes 3 405B (free)", fullName: "nousresearch/hermes-3-llama-3.1-405b:free", path: "nousresearch/hermes-3-llama-3.1-405b:free", backend: .openrouter, sizeBytes: nil),
            LocalModel(shortName: "Claude Opus 4", fullName: "anthropic/claude-opus-4", path: "anthropic/claude-opus-4", backend: .openrouter, sizeBytes: nil),
            LocalModel(shortName: "Claude Sonnet 4", fullName: "anthropic/claude-sonnet-4", path: "anthropic/claude-sonnet-4", backend: .openrouter, sizeBytes: nil),
        ]
    }

    /// Models that require upstream llama.cpp (no TurboQuant KV cache support yet)
    private static let upstreamOnlyPatterns = ["gemma-4", "gemma4"]

    /// Whether a GGUF model needs upstream llama-server instead of TurboQuant
    private func needsUpstreamLlama(_ model: LocalModel) -> Bool {
        let lower = model.shortName.lowercased()
        return Self.upstreamOnlyPatterns.contains { lower.contains($0) }
    }

    /// Whether a GGUF model supports TurboQuant KV cache compression
    func supportsTurboQuant(_ model: LocalModel) -> Bool {
        return model.backend == .gguf && !needsUpstreamLlama(model)
    }

    // MARK: - Smart Router local model

    /// Pick the best GGUF model for Smart Router's local backend.
    /// Prefers TurboQuant-compatible models (faster KV cache), then upstream GGUF.
    private func pickLocalModel() -> LocalModel? {
        let gguf = availableGGUFModels()
        // Prefer TurboQuant-compatible (Qwen etc.) for speed
        let turbo = gguf.filter { supportsTurboQuant($0) }
        if let best = turbo.max(by: { ($0.sizeBytes ?? 0) < ($1.sizeBytes ?? 0) }) {
            return best
        }
        // Fall back to any GGUF
        return gguf.max(by: { ($0.sizeBytes ?? 0) < ($1.sizeBytes ?? 0) })
    }

    /// Start a local model on the secondary port for Smart Router
    private func startLocalModel() -> Process? {
        guard let model = pickLocalModel() else {
            print("[smart] No GGUF model found for local backend")
            return nil
        }

        let useUpstream = needsUpstreamLlama(model)
        let binary = useUpstream ? Self.llamaServerUpstream : Self.llamaServerBinary

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: binary)

        var args = [
            "-m", model.path,
            "--alias", model.shortName,
            "--jinja", "-ngl", "99",
            "-c", "131072",  // 128K context for local model
            "-np", "1",
            "--host", "127.0.0.1",
            "--port", "\(Self.localModelPort)"
        ]
        if !useUpstream {
            args += ["-fa", "on", "--cache-type-k", "turbo3", "--cache-type-v", "turbo3"]
        } else {
            args += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
        }
        proc.arguments = args

        let logPath = "/tmp/smart-local-model.log"
        FileManager.default.createFile(atPath: logPath, contents: nil)
        let logHandle = FileHandle(forWritingAtPath: logPath)!
        proc.standardOutput = logHandle
        proc.standardError = logHandle

        print("[smart] Starting local model: \(model.shortName) on :\(Self.localModelPort) via \(useUpstream ? "upstream" : "turboquant")")

        do {
            try proc.run()
            return proc
        } catch {
            print("[smart] Failed to start local model: \(error)")
            return nil
        }
    }

    private func killLocalModel() {
        if let proc = localModelProcess, proc.isRunning {
            let pid = proc.processIdentifier
            proc.terminate()
            DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                if proc.isRunning { kill(pid, SIGKILL) }
            }
            proc.waitUntilExit()
        }
        localModelProcess = nil
        shell("pkill -f 'llama-server.*--port \(Self.localModelPort)' 2>/dev/null")
    }

    // MARK: - Start / Stop

    func serve(model: LocalModel) {
        if isLoading { return }
        isLoading = true

        state = .loading(progress: "Loading \(model.shortName)...")
        activeModelName = model.shortName
        activeBackend = model.backend
        turboQuantEnabled = model.backend == .gguf && !needsUpstreamLlama(model)
        modelSwitchTime = Date().timeIntervalSince1970
        liveTokS = nil
        healthFailCount = 0
        onStatusChange?()

        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }

            self.killServer()
            self.killLocalModel()

            // Wait until port is actually free (up to 10s)
            for _ in 0..<20 {
                let check = self.shell("lsof -ti tcp:\(self.port) 2>/dev/null")
                if check.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { break }
                usleep(500_000) // 0.5s
            }

            let proc = Process()
            let logPath: String
            switch model.backend {
            case .gguf: logPath = "/tmp/turboquant-server.log"
            case .openrouter: logPath = "/tmp/openrouter-proxy.log"
            case .smartrouter: logPath = "/tmp/smart-router.log"
            case .cloudrouter: logPath = "/tmp/cloud-router.log"
            default: logPath = Self.logFile
            }
            FileManager.default.createFile(atPath: logPath, contents: nil)
            let logHandle = FileHandle(forWritingAtPath: logPath)!

            switch model.backend {
            case .mlx:
                proc.executableURL = URL(fileURLWithPath: Self.uvBinary)
                proc.arguments = [
                    "run", "--with", "jang[mlx]", "--with", "mlx-lm",
                    "python3", "-u", Self.serveScript,
                    "--model", model.path,
                    "--model-alias", "local llm",
                    "--port", "\(self.port)"
                ]

            case .gguf:
                let useUpstream = self.needsUpstreamLlama(model)
                let binary = useUpstream ? Self.llamaServerUpstream : Self.llamaServerBinary
                proc.executableURL = URL(fileURLWithPath: binary)
                let ctxSize = "262144"  // 256K for all GGUF — KV is only 1.5GB even at full context
                var args = [
                    "-m", model.path,
                    "--alias", model.shortName,
                    "--jinja", "-ngl", "99",
                    "-c", ctxSize,
                    "-np", "1",
                    "--metrics",
                    "--host", "127.0.0.1",
                    "--port", "\(self.port)"
                ]
                if useUpstream {
                    // Upstream llama.cpp: FA + q4_0 KV cache
                    args += ["-fa", "on", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]
                } else {
                    // TurboQuant: FA + turbo3 KV compression (4.6x, enables 256K at speed)
                    args += ["-fa", "on", "--cache-type-k", "turbo3", "--cache-type-v", "turbo3"]
                }
                proc.arguments = args

            case .openrouter:
                let apiKey = Self.loadOpenRouterKey()
                proc.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
                proc.arguments = [
                    Self.proxyScript,
                    "\(self.port)",
                    model.path,  // OpenRouter model ID
                    apiKey
                ]

            case .smartrouter:
                // 1. Start local model on secondary port
                self.killLocalModel()
                for _ in 0..<10 {
                    let check = self.shell("lsof -ti tcp:\(Self.localModelPort) 2>/dev/null")
                    if check.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { break }
                    usleep(500_000)
                }
                self.localModelProcess = self.startLocalModel()

                // 2. Start smart proxy on main port
                let smartApiKey = Self.loadOpenRouterKey()
                proc.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
                proc.arguments = [
                    Self.smartProxyScript,
                    "\(self.port)",
                    "\(Self.localModelPort)",
                    smartApiKey
                ]

            case .cloudrouter:
                let cloudApiKey = Self.loadOpenRouterKey()
                proc.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
                proc.arguments = [
                    Self.cloudSmartProxyScript,
                    "\(self.port)",
                    cloudApiKey
                ]
            }

            proc.standardOutput = logHandle
            proc.standardError = logHandle

            proc.terminationHandler = { [weak self] p in
                DispatchQueue.main.async {
                    self?.serverProcess = nil
                    self?.isLoading = false
                    // SIGTERM (15) and SIGKILL (9) are normal during model switches — not crashes
                    let normalSignals: Set<Int32> = [0, 9, 15]
                    if !normalSignals.contains(p.terminationStatus), self?.state != .stopped {
                        self?.state = .crashed("Exit code \(p.terminationStatus)")
                        self?.onStatusChange?()
                    }
                }
            }

            do {
                try proc.run()
                DispatchQueue.main.async {
                    self.serverProcess = proc
                }
            } catch {
                DispatchQueue.main.async {
                    self.state = .crashed(error.localizedDescription)
                    self.isLoading = false
                    self.onStatusChange?()
                }
            }
        }
    }

    func stopServer() {
        isLoading = false
        killServer()
        killLocalModel()
        serverProcess = nil
        state = .stopped
        activeModelName = ""
        healthFailCount = 0
        onStatusChange?()
    }

    func restartCurrentModel() {
        killServer()
        sleep(1)
        state = .loading(progress: "Restarting...")
        healthFailCount = 0
        onStatusChange?()
        if activeBackend == .mlx {
            shell("launchctl stop \(Self.launchdLabel) 2>/dev/null; sleep 1; launchctl start \(Self.launchdLabel)")
        }
    }

    private static let launchdPlist = home
        .appendingPathComponent("Library/LaunchAgents/com.llm-router.server.plist").path
    private static let launchdLabel = "com.llm-router.server"

    private func killServer() {
        // Kill our tracked process first
        if let proc = serverProcess, proc.isRunning {
            let pid = proc.processIdentifier
            proc.terminate()
            // Give it 2s to exit gracefully before force-killing
            DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                if proc.isRunning { kill(pid, SIGKILL) }
            }
            proc.waitUntilExit()
        }
        serverProcess = nil
        // Unload launchd to prevent respawn, then clean up known server processes
        // Use specific process matching to avoid killing unrelated processes
        shell("""
            launchctl unload '\(Self.launchdPlist)' 2>/dev/null
            pkill -f 'serve_mlx.py.*--port \(self.port)' 2>/dev/null
            pkill -f 'openrouter-proxy.py.*\(self.port)' 2>/dev/null
            pkill -f 'smart-proxy.py.*\(self.port)' 2>/dev/null
            pkill -f 'cloud-smart-proxy.py.*\(self.port)' 2>/dev/null
            pkill -f 'llama-server.*--port \(self.port)' 2>/dev/null
            sleep 1
            """)
    }

    /// Re-enable launchd service (for MLX models that use it)
    private func enableLaunchd() {
        shell("launchctl load '\(Self.launchdPlist)' 2>/dev/null")
    }

    // MARK: - Health polling

    private func startHealthPolling() {
        healthTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            self?.pollSystemMetrics()
            self?.checkHealth()
            if case .ready = self?.state, self?.activeBackend == .gguf {
                self?.fetchLlamaMetrics()
            }
        }
        pollSystemMetrics()
        checkHealth()
    }

    /// Always poll CPU/MEM/GPU regardless of server state
    private func pollSystemMetrics() {
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            let memUsed = self.getMemoryUsed()
            let cpu = self.getCPUUsageDelta()
            let gpu = self.getGPUUsage()
            DispatchQueue.main.async {
                if let memUsed = memUsed { self.memUsedGB = memUsed }
                if let cpu = cpu { self.cpuUsage = cpu }
                if let gpu = gpu { self.gpuUsage = gpu }
                self.onStatusChange?()
            }
        }
    }

    private func checkHealth() {
        // Check server health via /health (works for both MLX and llama.cpp)
        let url = URL(string: "http://127.0.0.1:\(port)/health")!
        let request = URLRequest(url: url, timeoutInterval: 2)

        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                    self.healthFailCount = 0
                    self.isLoading = false

                    let wasReady: Bool
                    if case .ready = self.state { wasReady = true } else { wasReady = false }

                    self.state = .ready

                    // Detect backend and fetch details
                    self.detectAndFetchBackendDetails()

                    if !wasReady {
                        // Don't override name for routers — already set correctly
                        if self.activeBackend != .smartrouter && self.activeBackend != .cloudrouter {
                            self.fetchModelName()
                        }
                        self.onStatusChange?()
                    }
                    // Poll router stats when active
                    if self.activeBackend == .smartrouter || self.activeBackend == .cloudrouter {
                        self.fetchRouterStats()
                    }
                } else if let http = response as? HTTPURLResponse, http.statusCode == 503 {
                    self.healthFailCount = 0
                    if case .loading = self.state { return }
                    self.state = .loading(progress: "Server loading...")
                    self.onStatusChange?()
                } else {
                    self.healthFailCount += 1
                    if case .ready = self.state, self.healthFailCount >= 2 {
                        self.state = .crashed("Server stopped responding")
                        self.isLoading = false
                        self.onStatusChange?()
                    } else if case .loading = self.state, self.healthFailCount >= 10 {
                        self.state = .crashed("Server died during loading")
                        self.isLoading = false
                        self.onStatusChange?()
                    }
                }
            }
        }.resume()
    }

    /// Fetch routing stats from Smart/Cloud Router /stats endpoint
    private func fetchRouterStats() {
        let url = URL(string: "http://127.0.0.1:\(port)/stats")!
        let request = URLRequest(url: url, timeoutInterval: 2)

        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            guard let self = self,
                  let http = response as? HTTPURLResponse, http.statusCode == 200,
                  let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            DispatchQueue.main.async {
                self.routerRequests = json["total_requests"] as? Int ?? 0
                if let routing = json["routing"] as? [String: Any] {
                    // Smart Router has "local", Cloud Router has "qwen" — both are the cheap tier
                    let localData = (routing["local"] as? [String: Any]) ?? (routing["qwen"] as? [String: Any])
                    self.routerPctLocal = Int(localData?["pct"] as? Double ?? 0)
                }
                if let cost = json["cost"] as? [String: Any] {
                    self.routerSavedPct = cost["saving_pct"] as? Double ?? 0
                    self.routerCostActual = cost["actual"] as? Double ?? 0
                }
                self.onStatusChange?()
            }
        }.resume()
    }

    /// Auto-detect which backend is running and fetch appropriate metrics
    private func detectAndFetchBackendDetails() {
        // Routers and OpenRouter manage their own backend — don't override
        if activeBackend == .smartrouter || activeBackend == .cloudrouter || activeBackend == .openrouter {
            return
        }

        // Try /metrics first (llama.cpp only)
        let metricsURL = URL(string: "http://127.0.0.1:\(port)/metrics")!
        let metricsReq = URLRequest(url: metricsURL, timeoutInterval: 2)

        URLSession.shared.dataTask(with: metricsReq) { [weak self] data, response, _ in
            guard let self = self else { return }
            if let http = response as? HTTPURLResponse, http.statusCode == 200,
               let data = data, let text = String(data: data, encoding: .utf8),
               text.contains("llamacpp:") {
                // It's llama.cpp — don't override turboQuantEnabled (set correctly in serve())
                DispatchQueue.main.async {
                    self.activeBackend = .gguf
                }
                // Parse tok/s
                for line in text.components(separatedBy: .newlines) {
                    if line.hasPrefix("llamacpp:predicted_tokens_seconds ") {
                        let val = line.replacingOccurrences(of: "llamacpp:predicted_tokens_seconds ", with: "")
                        if let tokS = Double(val), tokS > 0 {
                            DispatchQueue.main.async {
                                self.liveTokS = tokS
                                self.onStatusChange?()
                            }
                        }
                        break
                    }
                }
            } else {
                // Not llama.cpp — try MLX /stats
                DispatchQueue.main.async {
                    self.turboQuantEnabled = false
                    self.activeBackend = .mlx
                }
                self.fetchMLXStats()
            }
        }.resume()
    }

    /// Fetch MLX-specific stats from /stats endpoint
    private func fetchMLXStats() {
        let url = URL(string: "http://127.0.0.1:\(port)/stats")!
        let request = URLRequest(url: url, timeoutInterval: 2)

        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            guard let self = self,
                  let http = response as? HTTPURLResponse, http.statusCode == 200,
                  let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            DispatchQueue.main.async {
                self.mlxActiveGB = json["mlx_active_gb"] as? Double ?? 0
                let displayName = json["model_display"] as? String
                    ?? (json["model"] as? String)?.split(separator: "/").last.map(String.init)
                    ?? self.activeModelName
                self.activeModelName = displayName
                self.onStatusChange?()
            }
        }.resume()
    }

    private func fetchModelName() {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/models")!
        let request = URLRequest(url: url, timeoutInterval: 2)

        URLSession.shared.dataTask(with: request) { [weak self] data, _, _ in
            guard let self = self, let data = data else { return }
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let models = json["data"] as? [[String: Any]],
               let first = models.first,
               let id = first["id"] as? String {
                DispatchQueue.main.async {
                    self.activeModelName = id
                    self.onStatusChange?()
                }
            }
        }.resume()
    }

    /// Fetch tok/s from llama.cpp /metrics endpoint
    private func fetchLlamaMetrics() {
        guard activeBackend == .gguf else { return }
        let url = URL(string: "http://127.0.0.1:\(port)/metrics")!
        let request = URLRequest(url: url, timeoutInterval: 2)

        URLSession.shared.dataTask(with: request) { [weak self] data, _, _ in
            guard let self = self, let data = data,
                  let text = String(data: data, encoding: .utf8) else { return }
            // Parse: llamacpp:predicted_tokens_seconds 73.1404
            for line in text.components(separatedBy: .newlines) {
                if line.hasPrefix("llamacpp:predicted_tokens_seconds ") {
                    let val = line.replacingOccurrences(of: "llamacpp:predicted_tokens_seconds ", with: "")
                    if let tokS = Double(val), tokS > 0 {
                        DispatchQueue.main.async {
                            self.liveTokS = tokS
                            self.onStatusChange?()
                        }
                    }
                    break
                }
            }
        }.resume()
    }

    // MARK: - Analytics

    private func readAnalytics() -> [[String: Any]]? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: Self.analyticsFile)),
              let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return nil }
        return arr
    }

    struct AnalyticsSummary {
        let requests24h: Int
        let avgTTFT24h: String
        let avgGen24h: String
        let requestsAll: Int
        let totalTokens: Int
        let avgTTFTAll: String
        let avgGenAll: String
        let lastTime: String
        let lastModel: String
        let lastTokensOut: Int
        let lastTokensIn: Int
        let lastTTFT: String
        let lastGenSpeed: String
    }

    func getAnalytics() -> AnalyticsSummary? {
        guard let data = readAnalytics(), !data.isEmpty else { return nil }

        let now = Date().timeIntervalSince1970
        let dayAgo = now - 86400

        let recent = data.filter { ($0["ts"] as? Double ?? 0) > dayAgo }

        func avgField(_ items: [[String: Any]], _ key: String) -> Double? {
            let vals = items.compactMap { ($0[key] as? Double).flatMap { $0 > 0 ? $0 : nil } }
            return vals.isEmpty ? nil : vals.reduce(0, +) / Double(vals.count)
        }

        let avgGen24h = avgField(recent, "gen_tok_s")
        let avgGenAll = avgField(data, "gen_tok_s")
        let avgTTFT24h = avgField(recent, "ttft_s")
        let avgTTFTAll = avgField(data, "ttft_s")
        let totalOut = data.reduce(0) { $0 + ($1["output_tokens"] as? Int ?? 0) }

        let last = data.last!
        let lastTs = last["ts"] as? Double ?? 0
        let df = DateFormatter()
        df.dateFormat = "HH:mm:ss"
        let lastTime = df.string(from: Date(timeIntervalSince1970: lastTs))
        let lastTTFT = last["ttft_s"] as? Double
        let lastGen = last["gen_tok_s"] as? Double ?? last["tok_s"] as? Double ?? 0

        return AnalyticsSummary(
            requests24h: recent.count,
            avgTTFT24h: avgTTFT24h.map { String(format: "%.2fs", $0) } ?? "n/a",
            avgGen24h: avgGen24h.map { String(format: "%.1f tok/s", $0) } ?? "n/a",
            requestsAll: data.count,
            totalTokens: totalOut,
            avgTTFTAll: avgTTFTAll.map { String(format: "%.2fs", $0) } ?? "n/a",
            avgGenAll: avgGenAll.map { String(format: "%.1f tok/s", $0) } ?? "n/a",
            lastTime: lastTime,
            lastModel: last["model"] as? String ?? "?",
            lastTokensOut: last["output_tokens"] as? Int ?? 0,
            lastTokensIn: last["input_tokens"] as? Int ?? 0,
            lastTTFT: lastTTFT.map { String(format: "%.2fs", $0) } ?? "n/a",
            lastGenSpeed: String(format: "%.1f tok/s", lastGen)
        )
    }

    // MARK: - Memory

    private func getMemoryUsed() -> Double? {
        let proc = Process()
        let pipe = Pipe()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/vm_stat")
        proc.standardOutput = pipe
        try? proc.run()
        proc.waitUntilExit()

        guard let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) else { return nil }

        var pages: [String: Int] = [:]
        for line in output.components(separatedBy: .newlines) {
            let pattern = #"Pages\s+([\w\s]+?):\s+(\d+)"#
            if let match = line.range(of: pattern, options: .regularExpression) {
                let parts = line[match].components(separatedBy: ":")
                if parts.count == 2 {
                    let key = parts[0].trimmingCharacters(in: .whitespaces).lowercased()
                    let val = Int(parts[1].trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ".", with: "")) ?? 0
                    pages[key] = val
                }
            }
        }

        let pageSize: Double = 16384
        let active = Double(pages["pages active"] ?? 0)
        let wired = Double(pages["pages wired down"] ?? 0)
        let compressed = Double(pages["pages occupied by compressor"] ?? 0)
        return (active + wired + compressed) * pageSize / 1e9
    }

    private func getCPUUsage() -> Int? {
        // Use host_statistics to get CPU load
        var loadInfo = host_cpu_load_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &loadInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }

        let user = Double(loadInfo.cpu_ticks.0)
        let system = Double(loadInfo.cpu_ticks.1)
        let idle = Double(loadInfo.cpu_ticks.2)
        let nice = Double(loadInfo.cpu_ticks.3)
        let total = user + system + idle + nice
        guard total > 0 else { return nil }
        return Int(((user + system + nice) / total) * 100)
    }

    private var prevCPUTicks: (user: Double, system: Double, idle: Double, nice: Double)?

    private func getCPUUsageDelta() -> Int? {
        var loadInfo = host_cpu_load_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &loadInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }

        let user = Double(loadInfo.cpu_ticks.0)
        let system = Double(loadInfo.cpu_ticks.1)
        let idle = Double(loadInfo.cpu_ticks.2)
        let nice = Double(loadInfo.cpu_ticks.3)

        defer {
            prevCPUTicks = (user, system, idle, nice)
        }

        guard let prev = prevCPUTicks else { return nil }

        let dUser = user - prev.user
        let dSystem = system - prev.system
        let dIdle = idle - prev.idle
        let dNice = nice - prev.nice
        let dTotal = dUser + dSystem + dIdle + dNice
        guard dTotal > 0 else { return nil }
        return Int(((dUser + dSystem + dNice) / dTotal) * 100)
    }

    private func getGPUUsage() -> Int? {
        let output = shell("ioreg -r -d 1 -c AGXAccelerator 2>/dev/null | grep 'Device Utilization'")
        // Parse: "Device Utilization %"=99
        guard let range = output.range(of: #"Device Utilization %"=(\d+)"#, options: .regularExpression) else { return nil }
        let match = String(output[range])
        guard let eqRange = match.range(of: "=") else { return nil }
        let numStr = match[eqRange.upperBound...]
        return Int(numStr)
    }

    // MARK: - Helpers

    @discardableResult
    private func shell(_ command: String) -> String {
        let proc = Process()
        let pipe = Pipe()
        proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
        proc.arguments = ["-c", command]
        proc.standardOutput = pipe
        proc.standardError = pipe
        try? proc.run()
        proc.waitUntilExit()
        return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }
}
