# Web Application - ML Visualizations

This web application uses Dioxus 0.6.0 to create interactive machine learning visualizations.

## Documentation

Comprehensive research and documentation on Dioxus capabilities:

- **[RESEARCH_SUMMARY.md](./RESEARCH_SUMMARY.md)** - Start here! Overview of all documentation and key findings
- **[DIOXUS_CAPABILITIES.md](./DIOXUS_CAPABILITIES.md)** - Complete guide to Dioxus 0.6 features (28KB)
- **[ML_VISUALIZATION_PATTERNS.md](./ML_VISUALIZATION_PATTERNS.md)** - Concrete ML visualization examples (39KB)
- **[DIOXUS_QUICK_REFERENCE.md](./DIOXUS_QUICK_REFERENCE.md)** - Quick lookup guide for common patterns (11KB)
- **[TESTING.md](./TESTING.md)** - Playwright testing setup and examples

## Project Structure

```
web/
├─ assets/             # Stylesheets and static assets
├─ src/
│  ├─ components/      # Reusable UI components
│  │  ├─ nav.rs        # Navigation component
│  │  ├─ view.rs       # View components
│  │  └─ mod.rs
│  ├─ routes.rs        # Route definitions
│  └─ main.rs          # Application entry point
├─ tests/              # Playwright browser tests
└─ Cargo.toml          # Dependencies and features
```

## Development

### Serving Your App

Run the following command in the root of your project to start developing:

```bash
dx serve                    # Web platform (default)
dx serve --platform desktop # Desktop platform
dx serve --platform mobile  # Mobile platform
```

### Building for Production

```bash
dx build --release
```

### Running Tests

```bash
npm test                    # Run Playwright tests
```

## Features

- **Cross-platform**: Web, desktop, and mobile from single codebase
- **Routing**: Navigation with Dioxus Router
- **Hot reload**: Automatic refresh on code changes
- **Type-safe**: Full Rust type system
- **Performant**: Fine-grained reactivity, optimized WASM

## Quick Start

1. Install Dioxus CLI:
   ```bash
   cargo install dioxus-cli
   ```

2. Start development server:
   ```bash
   dx serve
   ```

3. Open browser to `http://localhost:8080`

## Learn More

- [Dioxus Documentation](https://dioxuslabs.com/learn/0.6/guide)
- [API Reference](https://docs.rs/dioxus/latest/dioxus/)
- [Examples](https://github.com/DioxusLabs/dioxus/tree/main/examples)

