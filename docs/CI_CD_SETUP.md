# CI/CD Pipeline Documentation

## Overview

This project uses GitHub Actions for continuous integration and deployment. The CI/CD pipeline automatically validates all pull requests by running comprehensive tests across Rust code, WASM builds, and end-to-end browser tests.

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- All pull requests
- Pushes to `main` and `cicd` branches

**Jobs:**

#### `rust-tests`
Validates all Rust code quality and correctness.

**Steps:**
1. **Code Formatting:** Checks code follows Rust formatting standards (`cargo fmt`)
2. **Linting:** Runs Clippy with strict warnings (`cargo clippy`)
3. **Build:** Compiles all workspace packages
4. **Unit Tests:** Runs tests for all packages:
   - `linear_algebra` - Matrix and vector operations
   - `neural_network` - Neural network and optimizer tests (42 tests)
   - `linear_regression` - Gradient descent implementations
   - `loader` - Data I/O utilities

**Success Criteria:**
- All code must be formatted correctly
- No Clippy warnings allowed
- All tests must pass

#### `wasm-build`
Ensures the web application builds correctly for WASM.

**Steps:**
1. **Install Dioxus CLI:** Installs the web framework tooling
2. **Build WASM:** Compiles the web app for browser deployment
3. **Bundle Size Check:** Warns if WASM bundle exceeds 5MB
4. **Artifact Upload:** Saves compiled WASM for downstream jobs

**Success Criteria:**
- WASM build completes without errors
- Bundle size is reasonable (warning at 5MB+)

#### `playwright-tests`
Runs end-to-end browser tests to validate user-facing functionality.

**Dependencies:** Requires `wasm-build` to complete first

**Steps:**
1. **Setup Environment:** Installs Node.js, Playwright, and browsers
2. **Start Dev Server:** Launches the WASM app on localhost:8080
3. **Wait for Ready:** Polls until server responds (60-second timeout)
4. **Run E2E Tests:** Executes Playwright test suite
5. **Upload Reports:** Saves test results and HTML reports (30-day retention)

**Success Criteria:**
- Dev server starts successfully
- All Playwright tests pass
- Visual regression tests pass

#### `security-audit`
Scans dependencies for known security vulnerabilities.

**Steps:**
1. **Install cargo-audit:** Security auditing tool
2. **Scan Dependencies:** Checks Cargo.lock against RustSec advisory database

**Success Criteria:**
- No high or critical vulnerabilities found

#### `all-checks-complete`
Final validation that all jobs succeeded.

**Dependencies:** Requires all previous jobs

**Purpose:**
- Single status check for branch protection rules
- Provides clear pass/fail for the entire pipeline

---

### 2. Performance Benchmark Workflow (`.github/workflows/performance.yml`)

**Triggers:**
- Pull requests to `main` that modify:
  - `neural_network/**`
  - `linear_algebra/**`
  - `linear_regression/**`
  - `web/src/components/**`
- Pushes to `main`
- Manual workflow dispatch

**Jobs:**

#### `benchmark`
Runs performance benchmarks to detect regressions.

**Steps:**
1. **Run Optimizer Comparison:** Executes the optimizer benchmark example
2. **Capture Metrics:** Saves benchmark output
3. **Upload Results:** Archives for 90-day retention
4. **PR Comment:** Posts benchmark results directly on the pull request

**What It Measures:**
- Optimizer convergence speed (iterations to target)
- Allocation performance (zero-allocation validation)
- Execution time for 1000+ iterations

**Review Criteria:**
- No significant performance regressions vs. main branch
- Target: 1000+ iterations/sec maintained
- Zero allocations in hot paths preserved

---

### 3. PR Comment Workflow (`.github/workflows/pr-comment.yml`)

**Triggers:**
- Completion of CI workflow

**Permissions Required:**
- `pull-requests: write`
- `issues: write`

**Jobs:**

#### `comment-summary`
Posts a summary of all test results as a PR comment.

**Steps:**
1. **Download Artifacts:** Fetches test results from CI workflow
2. **Get PR Number:** Identifies the PR associated with the commit
3. **Post/Update Comment:** Creates or updates a summary comment

**Comment Format:**
```
## ✅ CI Test Results

**Status:** All checks passed!

### Test Summary
- Rust Tests: ✅ Passed
- WASM Build: ✅ Passed
- Playwright E2E: ✅ Passed
- Security Audit: ✅ Passed

[View detailed results](workflow_url)

🎉 All tests passed! This PR is ready for review.
```

**Behavior:**
- Updates existing comment (doesn't spam multiple comments)
- Shows clear pass/fail status with emojis
- Links to detailed workflow logs

---

## Branch Protection Rules

### Recommended Settings for `main` Branch

1. **Require status checks to pass before merging:**
   - `All Checks Complete` (from CI workflow)

2. **Require branches to be up to date before merging:** ✅ Enabled

3. **Require linear history:** ✅ Enabled (clean git history)

4. **Do not allow bypassing the above settings:** ✅ Enabled

### Setup Instructions

1. Go to repository **Settings** → **Branches**
2. Add rule for pattern: `main`
3. Enable: "Require status checks to pass before merging"
4. Search for and select: `All Checks Complete`
5. Enable: "Require branches to be up to date before merging"
6. Save changes

---

## Local Testing Before Pushing

### Run All Tests Locally

```bash
# Format check
cargo fmt --all -- --check

# Linting
cargo clippy --all-targets --all-features -- -D warnings

# Unit tests
cargo test --workspace

# WASM build
cd web && dx build --platform web --release

# Playwright tests (requires dev server running)
cd web
dx serve --port 8080 &
npx playwright test
```

### Quick Validation Script

Create `.github/scripts/pre-push.sh`:

```bash
#!/bin/bash
set -e

echo "🔍 Running pre-push checks..."

echo "📝 Checking formatting..."
cargo fmt --all -- --check

echo "🔎 Running Clippy..."
cargo clippy --all-targets --all-features -- -D warnings

echo "🧪 Running tests..."
cargo test --workspace

echo "📦 Building WASM..."
cd web && dx build --platform web --release && cd ..

echo "✅ All checks passed! Safe to push."
```

Make executable: `chmod +x .github/scripts/pre-push.sh`

---

## Performance Monitoring

### WASM Bundle Size Tracking

The CI workflow monitors WASM bundle size and warns if it exceeds 5MB.

**Current target:** < 2MB (compressed)

**If bundle grows:**
1. Check for accidentally included dependencies
2. Review `Cargo.toml` for unused features
3. Consider code splitting with `dyn` imports
4. Use `wasm-opt` for additional optimization

### Benchmark Tracking

Performance benchmarks run automatically on:
- PRs modifying core ML libraries
- PRs modifying web components
- All pushes to `main`

**Review checklist:**
- [ ] Optimizer iterations/sec ≥ 1000
- [ ] No increase in allocations in hot paths
- [ ] Benchmark completion time similar to baseline

---

## Troubleshooting

### CI Failures

#### `rust-tests` job fails

**Common causes:**
1. **Formatting issues:** Run `cargo fmt --all`
2. **Clippy warnings:** Run `cargo clippy --fix`
3. **Test failures:** Run `cargo test` locally to reproduce

#### `wasm-build` job fails

**Common causes:**
1. **Missing WASM target:** `rustup target add wasm32-unknown-unknown`
2. **Dioxus CLI version mismatch:** Check `Cargo.toml` compatibility
3. **Compile errors:** Test locally with `dx build --platform web`

#### `playwright-tests` job fails

**Common causes:**
1. **Dev server timeout:** Check if WASM build is slow (increase timeout)
2. **Test flakiness:** Review Playwright test selectors
3. **Visual regressions:** Update baseline screenshots if intentional

**Debug steps:**
```bash
# Run Playwright in headed mode locally
cd web
dx serve --port 8080 &
npx playwright test --headed
```

#### `security-audit` job fails

**Common causes:**
1. **Vulnerable dependency:** Update to patched version
2. **No patch available:** Consider alternative dependency or ignore with justification

**Fix vulnerable dependency:**
```bash
cargo update -p <vulnerable_package>
cargo audit
```

---

## Workflow Optimization

### Caching Strategy

The CI workflows use aggressive caching:

1. **Rust dependencies:** `Swatinem/rust-cache@v2`
   - Caches `target/` directory
   - Caches Cargo registry
   - Auto-invalidates on `Cargo.lock` changes

2. **Playwright browsers:** Cached in CI runner
   - 1.5GB+ of browser binaries
   - Persists across workflow runs

### Parallelization

Jobs run in parallel where possible:

```
rust-tests ─┐
            ├──> all-checks-complete
wasm-build ─┼──> playwright-tests ─┘
            │
security-audit ─┘
```

**Total pipeline time:** ~5-8 minutes (typical PR)

---

## Adding New Tests

### Adding Rust Unit Tests

1. Write tests in `<crate>/src/lib.rs` or `<crate>/tests/`
2. Tests automatically run in `rust-tests` job
3. No workflow changes needed

### Adding Playwright E2E Tests

1. Create test in `web/tests/<feature>.spec.ts`
2. Follow existing patterns for selectors
3. Test automatically discovered and run
4. Update baseline screenshots if adding visual tests:
   ```bash
   cd web
   npx playwright test --update-snapshots
   ```

### Adding Benchmarks

1. Create example in `<crate>/examples/benchmark_*.rs`
2. Update `performance.yml` to run new benchmark:
   ```yaml
   - name: Run new benchmark
     run: cargo run --release --example benchmark_new -p <crate>
   ```

---

## Maintenance

### Updating Workflow Actions

Check for updates quarterly:

```bash
# Update all actions to latest major versions
# Edit .github/workflows/*.yml and update:
# actions/checkout@v4 → actions/checkout@v5 (when available)
# dtolnay/rust-toolchain@stable (auto-updates)
# Swatinem/rust-cache@v2 → @v3
```

### Monitoring Costs

GitHub Actions minutes:
- **Free tier:** 2,000 minutes/month (public repos: unlimited)
- **Typical PR:** ~8 minutes total
- **Storage:** Artifacts auto-expire (7-90 days)

**Cost optimization:**
- Artifacts expire automatically
- Caching reduces build times
- Parallel jobs minimize wall-clock time

---

## Security Considerations

### Secrets Management

**Currently no secrets required.**

If adding deployment:
1. Use GitHub Secrets for sensitive data
2. Never commit API keys or tokens
3. Use `${{ secrets.NAME }}` in workflows

### Dependency Scanning

`cargo-audit` runs on every PR:
- Checks against RustSec Advisory Database
- Fails build on high/critical vulnerabilities
- Update vulnerable dependencies promptly

### WASM Security

- No `unsafe` code in web-facing paths
- Input validation on all CSV uploads
- No arbitrary code execution in browser

---

## Future Enhancements

### Planned Additions

1. **Code Coverage:** Add coverage reporting with `cargo-llvm-cov`
2. **Deployment:** Auto-deploy to GitHub Pages on `main` push
3. **Release Automation:** Create releases with `semantic-release`
4. **Benchmark Regression Detection:** Track performance over time
5. **Visual Regression Testing:** Automated screenshot comparison

### Integration Opportunities

- **Dependabot:** Auto-update dependencies
- **Renovate:** Alternative dependency management
- **CodeQL:** Advanced security scanning
- **Lighthouse CI:** Web performance metrics

---

## Support

For CI/CD issues:
1. Check workflow logs in GitHub Actions tab
2. Review this documentation
3. Test locally with provided scripts
4. Open issue with workflow run URL

---

**Last Updated:** November 8, 2025
**Status:** Production-ready CI/CD pipeline
**Maintainer:** RustML Team
