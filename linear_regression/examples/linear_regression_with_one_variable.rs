use linear_regression::LinearRegressor;
use loader::head;
use notes::execute;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let filepath = manifest_dir.join("data").join("car_data.csv");
    execute!(head(&filepath, 10).unwrap());
    let model = LinearRegressor::<f64>::new();
    println!("Model: {:?}", model);
    // filename, if the file has headers, training columns, target column
    model.fit_from_file(&filepath, true, vec![4], 2);
    // model.predict();

    // plot data and line
}
