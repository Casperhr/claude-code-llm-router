import SwiftUI
import AppKit

@main
struct LocalLLMApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings { EmptyView() }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var server = ServerManager()
    private var animationTimer: Timer?
    private var animationFrame: Int = 0
    private var titleUpdateTimer: Timer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        updateIcon()
        buildMenu()

        server.onStatusChange = { [weak self] in
            DispatchQueue.main.async {
                self?.updateIcon()
                self?.buildMenu()
            }
        }

        // Update title bar speed every 5s
        titleUpdateTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.updateTitleSpeed()
        }
    }

    private func updateIcon() {
        guard let button = statusItem.button else { return }
        button.toolTip = server.tooltip

        switch server.state {
        case .stopped:
            stopAnimation()
            button.image = NSImage(systemSymbolName: "sparkle", accessibilityDescription: "Stopped")
            button.image?.isTemplate = true
            button.contentTintColor = nil
            updateSystemStats()

        case .loading:
            startAnimation()
            button.title = ""
            button.attributedTitle = NSAttributedString(string: "")

        case .ready:
            stopAnimation()
            button.image = NSImage(systemSymbolName: "sparkle", accessibilityDescription: "Ready")
            button.image?.isTemplate = false
            button.contentTintColor = .systemCyan
            updateTitleSpeed()

        case .crashed:
            stopAnimation()
            button.image = NSImage(systemSymbolName: "sparkle", accessibilityDescription: "Crashed")
            button.image?.isTemplate = false
            button.contentTintColor = .systemRed
            updateSystemStats()
        }
    }

    private func updateTitleSpeed() {
        guard let button = statusItem.button, case .ready = server.state else { return }
        let title = NSMutableAttributedString()
        let dim: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor.white.withAlphaComponent(0.5),
            .font: NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .regular),
        ]
        let bright: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor.white,
            .font: NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .medium),
        ]
        let green: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor.systemGreen,
            .font: NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .medium),
        ]

        if server.activeBackend == .smartrouter || server.activeBackend == .cloudrouter {
            // Router mode: reqs, cost, savings, then hardware
            if server.routerRequests > 0 {
                title.append(NSAttributedString(string: " \(server.routerRequests)", attributes: bright))
                title.append(NSAttributedString(string: "r ", attributes: dim))
                title.append(NSAttributedString(string: "$\(String(format: "%.2f", server.routerCostActual))", attributes: bright))
                title.append(NSAttributedString(string: " ", attributes: dim))
                title.append(NSAttributedString(string: "-\(String(format: "%.0f", server.routerSavedPct))%", attributes: green))
            }
            title.append(NSAttributedString(string: "  ", attributes: dim))
            title.append(NSAttributedString(string: "\(Int(server.memUsedGB))", attributes: bright))
            title.append(NSAttributedString(string: "m ", attributes: dim))
            title.append(NSAttributedString(string: "\(server.cpuUsage)", attributes: bright))
            title.append(NSAttributedString(string: "c ", attributes: dim))
            title.append(NSAttributedString(string: "\(server.gpuUsage)", attributes: bright))
            title.append(NSAttributedString(string: "g", attributes: dim))
        } else {
            if let tokS = server.avgGenTokS {
                title.append(NSAttributedString(string: " \(String(format: "%.0f", tokS))", attributes: bright))
                title.append(NSAttributedString(string: "t/s", attributes: dim))
            }
            title.append(NSAttributedString(string: "  ", attributes: dim))
            title.append(NSAttributedString(string: "\(Int(server.memUsedGB))", attributes: bright))
            title.append(NSAttributedString(string: "m ", attributes: dim))
            title.append(NSAttributedString(string: "\(server.cpuUsage)", attributes: bright))
            title.append(NSAttributedString(string: "c ", attributes: dim))
            title.append(NSAttributedString(string: "\(server.gpuUsage)", attributes: bright))
            title.append(NSAttributedString(string: "g", attributes: dim))
        }

        button.title = ""
        button.attributedTitle = title
    }

    /// Show just system stats (no tok/s) — for stopped/crashed states
    private func updateSystemStats() {
        guard let button = statusItem.button else { return }
        let dim: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor.white.withAlphaComponent(0.4),
            .font: NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .regular),
        ]
        let bright: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor.white.withAlphaComponent(0.7),
            .font: NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .medium),
        ]
        let title = NSMutableAttributedString()
        title.append(NSAttributedString(string: " \(Int(server.memUsedGB))", attributes: bright))
        title.append(NSAttributedString(string: "m ", attributes: dim))
        title.append(NSAttributedString(string: "\(server.cpuUsage)", attributes: bright))
        title.append(NSAttributedString(string: "c ", attributes: dim))
        title.append(NSAttributedString(string: "\(server.gpuUsage)", attributes: bright))
        title.append(NSAttributedString(string: "g", attributes: dim))
        button.title = ""
        button.attributedTitle = title
    }

    // MARK: - Loading animation

    private let loadingFrames = ["sparkle", "sparkles", "sparkle", "sparkles"]

    private func startAnimation() {
        guard animationTimer == nil else { return }
        animationFrame = 0
        animationTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self = self, let button = self.statusItem.button else { return }
            let symbolName = self.loadingFrames[self.animationFrame % self.loadingFrames.count]
            button.image = NSImage(systemSymbolName: symbolName, accessibilityDescription: "Loading")
            button.image?.isTemplate = false
            button.contentTintColor = .systemOrange
            button.toolTip = self.server.tooltip
            self.animationFrame += 1
        }
    }

    private func stopAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
    }

    // MARK: - Header view

    private func makeHeaderView() -> NSView {
        let width: CGFloat = 260
        let padding: CGFloat = 16

        switch server.state {
        case .ready:
            let tokS = server.avgGenTokS

            // Model name — for routers, show savings on the first line
            let isRouter = server.activeBackend == .smartrouter || server.activeBackend == .cloudrouter
            let nameText: String
            if isRouter && server.routerRequests > 0 {
                nameText = "\(server.activeModelName)  ↓\(String(format: "%.0f", server.routerSavedPct))%  $\(String(format: "%.2f", server.routerCostActual))"
            } else {
                nameText = server.activeModelName
            }
            let nameLabel = NSTextField(labelWithString: nameText)
            nameLabel.font = .systemFont(ofSize: 13, weight: .semibold)
            nameLabel.textColor = .labelColor

            // Stats line
            let backendTag: String
            switch server.activeBackend {
            case .gguf: backendTag = server.turboQuantEnabled ? "TurboQuant" : "GGUF"
            case .mlx: backendTag = "MLX"
            case .openrouter: backendTag = "OpenRouter"
            case .smartrouter: backendTag = "Smart Router"
            case .cloudrouter: backendTag = "Cloud Router"
            case .ollama: backendTag = "Ollama"
            }
            var stats: [String] = [backendTag]
            if isRouter {
                if server.routerRequests > 0 {
                    stats.append("\(server.routerRequests) reqs")
                } else {
                    stats.append("ready")
                }
            } else {
                if server.mlxActiveGB > 0 { stats.append("\(String(format: "%.1f", server.mlxActiveGB)) GB") }
                if let t = tokS { stats.append("\(String(format: "%.0f", t)) tok/s") }
            }
            stats.append("CPU \(server.cpuUsage)%")
            stats.append("GPU \(server.gpuUsage)%")
            let statsLabel = NSTextField(labelWithString: stats.joined(separator: "  ·  "))
            statsLabel.font = .systemFont(ofSize: 11)
            statsLabel.textColor = .secondaryLabelColor

            // Green dot indicator
            let dot = NSView(frame: NSRect(x: 0, y: 0, width: 6, height: 6))
            dot.wantsLayer = true
            dot.layer?.backgroundColor = NSColor.systemGreen.cgColor
            dot.layer?.cornerRadius = 3

            let container = NSView(frame: NSRect(x: 0, y: 0, width: width, height: 52))

            for v in [dot, nameLabel, statsLabel] {
                v.translatesAutoresizingMaskIntoConstraints = false
                container.addSubview(v)
            }

            NSLayoutConstraint.activate([
                dot.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: padding),
                dot.centerYAnchor.constraint(equalTo: nameLabel.centerYAnchor),
                dot.widthAnchor.constraint(equalToConstant: 6),
                dot.heightAnchor.constraint(equalToConstant: 6),

                nameLabel.leadingAnchor.constraint(equalTo: dot.trailingAnchor, constant: 8),
                nameLabel.topAnchor.constraint(equalTo: container.topAnchor, constant: 8),
                nameLabel.trailingAnchor.constraint(lessThanOrEqualTo: container.trailingAnchor, constant: -padding),

                statsLabel.leadingAnchor.constraint(equalTo: nameLabel.leadingAnchor),
                statsLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 2),
                statsLabel.trailingAnchor.constraint(lessThanOrEqualTo: container.trailingAnchor, constant: -padding),
            ])

            return container

        case .loading(let progress):
            let nameLabel = NSTextField(labelWithString: progress)
            nameLabel.font = .systemFont(ofSize: 13, weight: .semibold)
            nameLabel.textColor = .systemOrange

            let sub: String
            if server.mlxActiveGB > 0 {
                sub = "\(String(format: "%.1f", server.mlxActiveGB)) GB loaded"
            } else {
                sub = "Starting server process..."
            }
            let subLabel = NSTextField(labelWithString: sub)
            subLabel.font = .systemFont(ofSize: 11)
            subLabel.textColor = .secondaryLabelColor

            // Orange dot
            let dot = NSView(frame: NSRect(x: 0, y: 0, width: 6, height: 6))
            dot.wantsLayer = true
            dot.layer?.backgroundColor = NSColor.systemOrange.cgColor
            dot.layer?.cornerRadius = 3

            let container = NSView(frame: NSRect(x: 0, y: 0, width: width, height: 52))

            for v in [dot, nameLabel, subLabel] {
                v.translatesAutoresizingMaskIntoConstraints = false
                container.addSubview(v)
            }

            NSLayoutConstraint.activate([
                dot.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: padding),
                dot.centerYAnchor.constraint(equalTo: nameLabel.centerYAnchor),
                dot.widthAnchor.constraint(equalToConstant: 6),
                dot.heightAnchor.constraint(equalToConstant: 6),

                nameLabel.leadingAnchor.constraint(equalTo: dot.trailingAnchor, constant: 8),
                nameLabel.topAnchor.constraint(equalTo: container.topAnchor, constant: 8),
                nameLabel.trailingAnchor.constraint(lessThanOrEqualTo: container.trailingAnchor, constant: -padding),

                subLabel.leadingAnchor.constraint(equalTo: nameLabel.leadingAnchor),
                subLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 2),
                subLabel.trailingAnchor.constraint(lessThanOrEqualTo: container.trailingAnchor, constant: -padding),
            ])

            return container

        case .crashed(let msg):
            let nameLabel = NSTextField(labelWithString: "Server crashed")
            nameLabel.font = .systemFont(ofSize: 13, weight: .semibold)
            nameLabel.textColor = .systemRed

            let subLabel = NSTextField(labelWithString: msg)
            subLabel.font = .systemFont(ofSize: 11)
            subLabel.textColor = .secondaryLabelColor

            let dot = NSView(frame: NSRect(x: 0, y: 0, width: 6, height: 6))
            dot.wantsLayer = true
            dot.layer?.backgroundColor = NSColor.systemRed.cgColor
            dot.layer?.cornerRadius = 3

            let container = NSView(frame: NSRect(x: 0, y: 0, width: width, height: 52))

            for v in [dot, nameLabel, subLabel] {
                v.translatesAutoresizingMaskIntoConstraints = false
                container.addSubview(v)
            }

            NSLayoutConstraint.activate([
                dot.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: padding),
                dot.centerYAnchor.constraint(equalTo: nameLabel.centerYAnchor),
                dot.widthAnchor.constraint(equalToConstant: 6),
                dot.heightAnchor.constraint(equalToConstant: 6),

                nameLabel.leadingAnchor.constraint(equalTo: dot.trailingAnchor, constant: 8),
                nameLabel.topAnchor.constraint(equalTo: container.topAnchor, constant: 8),
                nameLabel.trailingAnchor.constraint(lessThanOrEqualTo: container.trailingAnchor, constant: -padding),

                subLabel.leadingAnchor.constraint(equalTo: nameLabel.leadingAnchor),
                subLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 2),
                subLabel.trailingAnchor.constraint(lessThanOrEqualTo: container.trailingAnchor, constant: -padding),
            ])

            return container

        case .stopped:
            let label = NSTextField(labelWithString: "Select a model to start")
            label.font = .systemFont(ofSize: 13)
            label.textColor = .tertiaryLabelColor

            let dot = NSView(frame: NSRect(x: 0, y: 0, width: 6, height: 6))
            dot.wantsLayer = true
            dot.layer?.backgroundColor = NSColor.tertiaryLabelColor.cgColor
            dot.layer?.cornerRadius = 3

            let container = NSView(frame: NSRect(x: 0, y: 0, width: width, height: 36))

            for v in [dot, label] {
                v.translatesAutoresizingMaskIntoConstraints = false
                container.addSubview(v)
            }

            NSLayoutConstraint.activate([
                dot.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: padding),
                dot.centerYAnchor.constraint(equalTo: label.centerYAnchor),
                dot.widthAnchor.constraint(equalToConstant: 6),
                dot.heightAnchor.constraint(equalToConstant: 6),

                label.leadingAnchor.constraint(equalTo: dot.trailingAnchor, constant: 8),
                label.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            ])

            return container
        }
    }

    // MARK: - Menu

    private func buildMenu() {
        let menu = NSMenu()

        // Status header — custom NSView for full-color rendering
        let headerItem = NSMenuItem()
        headerItem.view = makeHeaderView()
        menu.addItem(headerItem)

        menu.addItem(.separator())

        // GGUF models — split by TurboQuant vs upstream
        let ggufModels = server.availableGGUFModels()
        let turboModels = ggufModels.filter { server.supportsTurboQuant($0) }
        let upstreamModels = ggufModels.filter { !server.supportsTurboQuant($0) }

        if !turboModels.isEmpty {
            let label = NSMenuItem(title: "TurboQuant (GGUF)", action: nil, keyEquivalent: "")
            label.isEnabled = false
            menu.addItem(label)

            for model in turboModels {
                let sizeStr = model.sizeLabel.isEmpty ? "" : "  \(model.sizeLabel)"
                let item = NSMenuItem(title: "  \(model.shortName)\(sizeStr)", action: #selector(selectModel(_:)), keyEquivalent: "")
                item.target = self
                item.representedObject = model.path
                if server.activeModelName == model.shortName && server.activeBackend == .gguf {
                    item.state = .on
                }
                menu.addItem(item)
            }

            menu.addItem(.separator())
        }

        if !upstreamModels.isEmpty {
            let label = NSMenuItem(title: "GGUF", action: nil, keyEquivalent: "")
            label.isEnabled = false
            menu.addItem(label)

            for model in upstreamModels {
                let sizeStr = model.sizeLabel.isEmpty ? "" : "  \(model.sizeLabel)"
                let item = NSMenuItem(title: "  \(model.shortName)\(sizeStr)", action: #selector(selectModel(_:)), keyEquivalent: "")
                item.target = self
                item.representedObject = model.path
                if server.activeModelName == model.shortName && server.activeBackend == .gguf {
                    item.state = .on
                }
                menu.addItem(item)
            }

            menu.addItem(.separator())
        }

        // MLX models
        let mlxModels = server.availableMLXModels()
        let mlxLabel = NSMenuItem(title: "MLX", action: nil, keyEquivalent: "")
        mlxLabel.isEnabled = false
        menu.addItem(mlxLabel)

        for model in mlxModels {
            let item = NSMenuItem(title: "  \(model.shortName)", action: #selector(selectModel(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = model.path

            if (server.activeModelName == model.shortName ||
                server.activeModelName == model.fullName.split(separator: "/").last.map(String.init))
                && !server.turboQuantEnabled {
                item.state = .on
            }
            menu.addItem(item)
        }

        menu.addItem(.separator())

        // Ollama models
        let ollamaModels = server.availableOllamaModels()
        if !ollamaModels.isEmpty {
            let ollamaLabel = NSMenuItem(title: "Ollama (local)", action: nil, keyEquivalent: "")
            ollamaLabel.isEnabled = false
            menu.addItem(ollamaLabel)

            for model in ollamaModels {
                let sizeStr = model.sizeLabel.isEmpty ? "" : "  \(model.sizeLabel)"
                let item = NSMenuItem(title: "  \(model.shortName)\(sizeStr)", action: #selector(selectModel(_:)), keyEquivalent: "")
                item.target = self
                item.representedObject = model.path
                if server.activeModelName == model.shortName && server.activeBackend == .ollama {
                    item.state = .on
                }
                menu.addItem(item)
            }

            menu.addItem(.separator())
        }

        // Smart Router
        let smartModels = server.availableSmartRouterModels()
        let smartLabel = NSMenuItem(title: "Smart Router", action: nil, keyEquivalent: "")
        smartLabel.isEnabled = false
        menu.addItem(smartLabel)

        for model in smartModels {
            let desc: String
            switch model.backend {
            case .smartrouter: desc = "local + Sonnet + Opus"
            case .cloudrouter: desc = "Qwen + Flash + Sonnet + Opus"
            default: desc = ""
            }
            let item = NSMenuItem(title: "  \(model.shortName)  (\(desc))", action: #selector(selectModel(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = model.path
            if server.activeBackend == model.backend { item.state = .on }
            menu.addItem(item)
        }

        menu.addItem(.separator())

        // OpenRouter models
        let orModels = server.availableOpenRouterModels()
        let orLabel = NSMenuItem(title: "OpenRouter", action: nil, keyEquivalent: "")
        orLabel.isEnabled = false
        menu.addItem(orLabel)

        for model in orModels {
            let item = NSMenuItem(title: "  \(model.shortName)", action: #selector(selectModel(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = model.path

            if server.activeModelName == model.shortName && server.activeBackend == .openrouter {
                item.state = .on
            }
            menu.addItem(item)
        }

        menu.addItem(.separator())

        // Actions
        if case .ready = server.state {
            let restart = NSMenuItem(title: "Restart Server", action: #selector(restartServer), keyEquivalent: "r")
            restart.target = self
            menu.addItem(restart)
        }

        if case .loading = server.state {
            // Show stop during loading
        }

        if case .ready = server.state {
            let stop = NSMenuItem(title: "Stop Server", action: #selector(stopServer), keyEquivalent: "s")
            stop.target = self
            menu.addItem(stop)
        } else if case .loading = server.state {
            let stop = NSMenuItem(title: "Stop Server", action: #selector(stopServer), keyEquivalent: "s")
            stop.target = self
            menu.addItem(stop)
        }

        menu.addItem(.separator())

        // Analytics
        let analyticsItem = NSMenuItem(title: "Analytics", action: #selector(showAnalytics), keyEquivalent: "a")
        analyticsItem.target = self
        menu.addItem(analyticsItem)

        // Copy endpoint
        let copyItem = NSMenuItem(title: "Copy Endpoint URL", action: #selector(copyEndpoint), keyEquivalent: "c")
        copyItem.target = self
        menu.addItem(copyItem)

        // View log
        let logItem = NSMenuItem(title: "View Server Log", action: #selector(openLog), keyEquivalent: "l")
        logItem.target = self
        menu.addItem(logItem)

        // Refresh
        let refreshItem = NSMenuItem(title: "Refresh Models", action: #selector(refreshModels), keyEquivalent: "f")
        refreshItem.target = self
        menu.addItem(refreshItem)

        menu.addItem(.separator())

        let quitItem = NSMenuItem(title: "Quit", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        self.statusItem.menu = menu
    }

    // MARK: - Actions

    @objc private func selectModel(_ sender: NSMenuItem) {
        guard let path = sender.representedObject as? String else { return }
        let allModels = server.availableGGUFModels() + server.availableMLXModels() + server.availableOllamaModels() + server.availableSmartRouterModels() + server.availableOpenRouterModels()
        guard let model = allModels.first(where: { $0.path == path }) else { return }
        server.serve(model: model)
    }

    @objc private func restartServer() {
        server.restartCurrentModel()
    }

    @objc private func stopServer() {
        server.stopServer()
    }

    @objc private func showAnalytics() {
        guard let stats = server.getAnalytics() else {
            let alert = NSAlert()
            alert.messageText = "Analytics"
            alert.informativeText = "No data yet — run some requests first."
            alert.runModal()
            return
        }

        let msg = """
        ━━ Last 24h ━━━━━━━━━━━━━━━━━━━━━━━
        Requests:   \(stats.requests24h)
        Avg TTFT:   \(stats.avgTTFT24h)
        Avg gen:    \(stats.avgGen24h)

        ━━ All time ━━━━━━━━━━━━━━━━━━━━━━━
        Requests:   \(stats.requestsAll)  (\(stats.totalTokens.formatted()) tokens)
        Avg TTFT:   \(stats.avgTTFTAll)
        Avg gen:    \(stats.avgGenAll)

        ━━ Last request ━━━━━━━━━━━━━━━━━━━
        Time:       \(stats.lastTime)
        Model:      \(stats.lastModel)
        Tokens:     \(stats.lastTokensOut) out  /  \(stats.lastTokensIn) in
        TTFT:       \(stats.lastTTFT)
        Gen speed:  \(stats.lastGenSpeed)
        """

        let alert = NSAlert()
        alert.messageText = "Analytics"
        alert.informativeText = msg
        alert.addButton(withTitle: "Close")
        alert.addButton(withTitle: "Copy to Clipboard")

        if alert.runModal() == .alertSecondButtonReturn {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(msg, forType: .string)
        }
    }

    @objc private func copyEndpoint() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("http://localhost:\(server.port)/v1", forType: .string)
    }

    @objc private func refreshModels() {
        buildMenu()
    }

    @objc private func openLog() {
        NSWorkspace.shared.open(URL(fileURLWithPath: "/tmp/mlx-server.log"))
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }
}
