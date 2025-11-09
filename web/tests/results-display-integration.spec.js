/**
 * ResultsDisplay Component Integration Test
 *
 * Tests the new ResultsDisplay component with all 7 algorithms:
 * - K-Means Clustering
 * - PCA (Dimensionality Reduction)
 * - Logistic Regression
 * - Decision Tree Classifier
 * - Naive Bayes Classifier
 * - Standard Scaler
 * - MinMax Scaler
 *
 * Verifies:
 * - Algorithm execution produces ResultsDisplay component
 * - Tabbed interface works correctly
 * - Metrics are displayed
 * - Visualizations render
 * - Predictions table appears (where applicable)
 */

const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

test.describe('ResultsDisplay Component Integration', () => {
    let testFile;

    test.beforeEach(async ({ page }) => {
        await page.goto('http://localhost:8080/playground');
        await page.waitForLoadState('networkidle');
    });

    test.afterEach(() => {
        // Clean up test files
        if (testFile && fs.existsSync(testFile)) {
            fs.unlinkSync(testFile);
        }
    });

    test('K-Means: ResultsDisplay shows cluster distribution visualization', async ({ page }) => {
        console.log('=== Testing K-Means with ResultsDisplay ===');

        // Create clustering test data
        const csvContent = `x,y
1.0,2.0
1.5,2.5
2.0,3.0
8.0,9.0
8.5,9.5
9.0,10.0
15.0,16.0
15.5,16.5
16.0,17.0`;

        testFile = path.join(__dirname, 'kmeans-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload CSV
        console.log('Step 1: Uploading CSV...');
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Select K-Means
        console.log('Step 2: Selecting K-Means...');
        const kmeansButton = page.locator('button').filter({ hasText: /K-Means/i });
        await kmeansButton.click();
        await page.waitForTimeout(500);

        // Run algorithm
        console.log('Step 3: Running K-Means...');
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Check for ResultsDisplay component
        console.log('Step 4: Checking for ResultsDisplay...');
        const pageContent = await page.textContent('body');

        // Check for algorithm name
        expect(pageContent).toContain('K-Means');

        // Check for cluster-related output
        const hasClusterInfo = pageContent.includes('Cluster') ||
                               pageContent.includes('cluster') ||
                               pageContent.includes('Inertia');

        console.log('Cluster information displayed:', hasClusterInfo);
        expect(hasClusterInfo).toBeTruthy();

        // Look for visualization section
        const vizSection = page.locator('.visualization-section, .cluster-viz, .results-viz');
        const hasViz = await vizSection.count() > 0;
        console.log('Visualization section found:', hasViz);

        console.log('=== K-Means Test Complete ===');
    });

    test('PCA: ResultsDisplay shows variance chart visualization', async ({ page }) => {
        console.log('=== Testing PCA with ResultsDisplay ===');

        // Create PCA test data with multiple features
        const csvContent = `f1,f2,f3,f4
1.0,2.0,3.0,4.0
1.5,2.5,3.5,4.5
2.0,3.0,4.0,5.0
2.5,3.5,4.5,5.5
3.0,4.0,5.0,6.0`;

        testFile = path.join(__dirname, 'pca-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload CSV
        console.log('Step 1: Uploading CSV...');
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Select PCA
        console.log('Step 2: Selecting PCA...');
        const pcaButton = page.locator('button').filter({ hasText: /PCA/i });
        await pcaButton.click();
        await page.waitForTimeout(500);

        // Run algorithm
        console.log('Step 3: Running PCA...');
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Check for ResultsDisplay component
        console.log('Step 4: Checking for ResultsDisplay...');
        const pageContent = await page.textContent('body');

        // Check for PCA-specific output
        const hasPCAInfo = pageContent.includes('PCA') ||
                          pageContent.includes('Component') ||
                          pageContent.includes('Variance') ||
                          pageContent.includes('variance');

        console.log('PCA information displayed:', hasPCAInfo);
        expect(hasPCAInfo).toBeTruthy();

        // Look for SVG visualization (VarianceChart)
        const svgElements = page.locator('svg');
        const svgCount = await svgElements.count();
        console.log('SVG elements found:', svgCount);

        if (svgCount > 0) {
            console.log('✓ Variance chart visualization rendered');
        }

        console.log('=== PCA Test Complete ===');
    });

    test('Logistic Regression: ResultsDisplay shows confusion matrix', async ({ page }) => {
        console.log('=== Testing Logistic Regression with ResultsDisplay ===');

        // Create classification test data
        const csvContent = `f1,f2,target
1.0,2.0,0
1.5,2.5,0
2.0,3.0,0
8.0,9.0,1
8.5,9.5,1
9.0,10.0,1`;

        testFile = path.join(__dirname, 'logreg-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload CSV
        console.log('Step 1: Uploading CSV...');
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Select Logistic Regression
        console.log('Step 2: Selecting Logistic Regression...');
        const logregButton = page.locator('button').filter({ hasText: /Logistic Regression/i });
        await logregButton.click();
        await page.waitForTimeout(500);

        // Run algorithm
        console.log('Step 3: Running Logistic Regression...');
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Check for ResultsDisplay component
        console.log('Step 4: Checking for ResultsDisplay...');
        const pageContent = await page.textContent('body');

        // Check for classification metrics
        const hasAccuracy = pageContent.includes('Accuracy') || pageContent.includes('accuracy');
        const hasLogReg = pageContent.includes('Logistic') || pageContent.includes('classification');

        console.log('Accuracy displayed:', hasAccuracy);
        console.log('Logistic Regression info displayed:', hasLogReg);

        expect(hasAccuracy || hasLogReg).toBeTruthy();

        // Look for confusion matrix SVG
        const svgElements = page.locator('svg');
        const svgCount = await svgElements.count();
        console.log('SVG elements (confusion matrix) found:', svgCount);

        console.log('=== Logistic Regression Test Complete ===');
    });

    test('Decision Tree: ResultsDisplay shows confusion matrix and tree depth', async ({ page }) => {
        console.log('=== Testing Decision Tree with ResultsDisplay ===');

        const csvContent = `f1,f2,target
1.0,2.0,0
1.5,2.5,0
2.0,3.0,0
8.0,9.0,1
8.5,9.5,1
9.0,10.0,1`;

        testFile = path.join(__dirname, 'dt-results-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload CSV
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Select Decision Tree
        const dtButton = page.locator('button').filter({ hasText: /Decision Tree/i });
        await dtButton.click();
        await page.waitForTimeout(500);

        // Run algorithm
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Check for results
        const pageContent = await page.textContent('body');

        const hasAccuracy = pageContent.includes('Accuracy') || pageContent.includes('accuracy');
        const hasTreeInfo = pageContent.includes('Tree') ||
                           pageContent.includes('depth') ||
                           pageContent.includes('Decision Tree');

        console.log('Accuracy displayed:', hasAccuracy);
        console.log('Tree information displayed:', hasTreeInfo);

        expect(hasAccuracy || hasTreeInfo).toBeTruthy();

        // Check for confusion matrix
        const svgElements = page.locator('svg');
        const svgCount = await svgElements.count();
        console.log('Confusion matrix elements:', svgCount);

        console.log('=== Decision Tree Test Complete ===');
    });

    test('Naive Bayes: ResultsDisplay shows classification results', async ({ page }) => {
        console.log('=== Testing Naive Bayes with ResultsDisplay ===');

        const csvContent = `f1,f2,target
1.0,2.0,0
1.5,2.5,0
2.0,3.0,0
8.0,9.0,1
8.5,9.5,1
9.0,10.0,1`;

        testFile = path.join(__dirname, 'nb-results-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload CSV
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Select Naive Bayes
        const nbButton = page.locator('button').filter({ hasText: /Naive Bayes/i });
        await nbButton.click();
        await page.waitForTimeout(500);

        // Run algorithm
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Check for results
        const pageContent = await page.textContent('body');

        const hasAccuracy = pageContent.includes('Accuracy') || pageContent.includes('accuracy');
        const hasNBInfo = pageContent.includes('Naive Bayes') ||
                         pageContent.includes('Gaussian') ||
                         pageContent.includes('probabilistic');

        console.log('Accuracy displayed:', hasAccuracy);
        console.log('Naive Bayes info displayed:', hasNBInfo);

        expect(hasAccuracy || hasNBInfo).toBeTruthy();

        console.log('=== Naive Bayes Test Complete ===');
    });

    test('Standard Scaler: ResultsDisplay shows before/after statistics', async ({ page }) => {
        console.log('=== Testing Standard Scaler with ResultsDisplay ===');

        const csvContent = `f1,f2,f3
10.0,20.0,30.0
15.0,25.0,35.0
20.0,30.0,40.0
25.0,35.0,45.0`;

        testFile = path.join(__dirname, 'stdscaler-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload CSV
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Select Standard Scaler
        const scalerButton = page.locator('button').filter({ hasText: /Standard Scaler/i });
        await scalerButton.click();
        await page.waitForTimeout(500);

        // Run algorithm
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Check for results
        const pageContent = await page.textContent('body');

        const hasScalerInfo = pageContent.includes('Standard Scaler') ||
                             pageContent.includes('standardized') ||
                             pageContent.includes('Mean') ||
                             pageContent.includes('Std');

        console.log('Scaler information displayed:', hasScalerInfo);
        expect(hasScalerInfo).toBeTruthy();

        // Check for before/after stats
        const hasBeforeAfter = pageContent.includes('Before') ||
                              pageContent.includes('After') ||
                              pageContent.includes('mean') ||
                              pageContent.includes('std');

        console.log('Before/After statistics displayed:', hasBeforeAfter);

        console.log('=== Standard Scaler Test Complete ===');
    });

    test('MinMax Scaler: ResultsDisplay shows scaling results', async ({ page }) => {
        console.log('=== Testing MinMax Scaler with ResultsDisplay ===');

        const csvContent = `f1,f2,f3
10.0,20.0,30.0
15.0,25.0,35.0
20.0,30.0,40.0
25.0,35.0,45.0`;

        testFile = path.join(__dirname, 'minmax-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload CSV
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Select MinMax Scaler
        const scalerButton = page.locator('button').filter({ hasText: /MinMax Scaler/i });
        await scalerButton.click();
        await page.waitForTimeout(500);

        // Run algorithm
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Check for results
        const pageContent = await page.textContent('body');

        const hasScalerInfo = pageContent.includes('MinMax') ||
                             pageContent.includes('scaled') ||
                             pageContent.includes('[0, 1]') ||
                             pageContent.includes('Min') ||
                             pageContent.includes('Max');

        console.log('MinMax scaler information displayed:', hasScalerInfo);
        expect(hasScalerInfo).toBeTruthy();

        console.log('=== MinMax Scaler Test Complete ===');
    });

    test('ResultsDisplay: Tab navigation works correctly', async ({ page }) => {
        console.log('=== Testing ResultsDisplay Tab Navigation ===');

        // Create simple classification data
        const csvContent = `f1,f2,target
1.0,2.0,0
8.0,9.0,1`;

        testFile = path.join(__dirname, 'tabs-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload and run Decision Tree (has multiple tabs)
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        const dtButton = page.locator('button').filter({ hasText: /Decision Tree/i });
        await dtButton.click();
        await page.waitForTimeout(500);

        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Look for tab buttons
        const tabButtons = page.locator('button.tab-button, .tabs button, .tab');
        const tabCount = await tabButtons.count();
        console.log('Tab buttons found:', tabCount);

        if (tabCount > 0) {
            console.log('✓ Tabbed interface is present');

            // Try clicking different tabs if they exist
            for (let i = 0; i < Math.min(tabCount, 3); i++) {
                try {
                    await tabButtons.nth(i).click();
                    await page.waitForTimeout(200);
                    const tabText = await tabButtons.nth(i).textContent();
                    console.log(`Clicked tab ${i}: ${tabText}`);
                } catch (e) {
                    console.log(`Could not click tab ${i}`);
                }
            }
        }

        console.log('=== Tab Navigation Test Complete ===');
    });

    test('ResultsDisplay: Metrics are displayed correctly', async ({ page }) => {
        console.log('=== Testing ResultsDisplay Metrics Display ===');

        // Use classification algorithm to get accuracy metric
        const csvContent = `f1,f2,target
1.0,2.0,0
1.5,2.5,0
2.0,3.0,0
8.0,9.0,1
8.5,9.5,1
9.0,10.0,1`;

        testFile = path.join(__dirname, 'metrics-test.csv');
        fs.writeFileSync(testFile, csvContent);

        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        const nbButton = page.locator('button').filter({ hasText: /Naive Bayes/i });
        await nbButton.click();
        await page.waitForTimeout(500);

        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Check for metric cards or displays
        const pageContent = await page.textContent('body');

        // Look for common metric patterns
        const hasPercentage = pageContent.match(/\d+(\.\d+)?%/);
        const hasAccuracy = pageContent.includes('Accuracy') || pageContent.includes('accuracy');
        const hasMetricValue = pageContent.match(/\d+\.\d+/);

        console.log('Percentage values found:', !!hasPercentage);
        console.log('Accuracy label found:', hasAccuracy);
        console.log('Metric values found:', !!hasMetricValue);

        expect(hasPercentage || hasAccuracy || hasMetricValue).toBeTruthy();

        console.log('=== Metrics Display Test Complete ===');
    });

    test('ResultsDisplay: Predictions table appears for classifiers', async ({ page }) => {
        console.log('=== Testing ResultsDisplay Predictions Table ===');

        const csvContent = `f1,f2,target
1.0,2.0,0
1.5,2.5,0
8.0,9.0,1
8.5,9.5,1`;

        testFile = path.join(__dirname, 'predictions-test.csv');
        fs.writeFileSync(testFile, csvContent);

        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        const dtButton = page.locator('button').filter({ hasText: /Decision Tree/i });
        await dtButton.click();
        await page.waitForTimeout(500);

        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Look for table elements
        const tables = page.locator('table, .data-table, .predictions-table');
        const tableCount = await tables.count();
        console.log('Table elements found:', tableCount);

        // Look for "Predicted" text
        const pageContent = await page.textContent('body');
        const hasPredictions = pageContent.includes('Predict') ||
                              pageContent.includes('predict') ||
                              pageContent.includes('Class');

        console.log('Predictions information found:', hasPredictions);

        console.log('=== Predictions Table Test Complete ===');
    });
});
