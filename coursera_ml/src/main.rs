use loader::read;
pub use notes::execute;

execute! {
    fn add_bug(a: i32, b: i32) -> i32 {
        let bug = 10;
        a + b * bug
    }
}

/// Sample function demonstrating docstrings [`execute`]
fn foo() {}

fn main() {
    let filepath = std::env::args().nth(1).expect("No filepath given");

    // read housing data
    if let Ok(lines) = read(&filepath) {
        for (i, line) in lines.map_while(Result::ok).enumerate() {
            if i >= 5 {
                break;
            }
            println!("{line}");
        }
    } else {
        println!("Could not open file {filepath}");
    }

    // explain linear regression
    // if let Ok(lines) = read(
    //

    execute!({
        let a = 10;
        let b = 5;
        a + b * 100
    });
    execute!(add_bug::call(1, 20));
    println!("{}", add_bug::SOURCE_CODE);
}
