//! A note-taking program for reading and writing notes.

use std::io::prelude::*;
use std::{fs::File, path::Path};

pub enum WriteModes {
    W,
    A,
}

pub fn write_note<P>(filepath: P, notes: &str, mode: WriteModes)
where
    P: AsRef<Path> + std::fmt::Debug,
{
    match mode {
        WriteModes::W => {
            println!("Writing to {:?}", filepath);
            let mut file = match File::create(&filepath) {
                Err(why) => panic!("Couldn't create {:?}: {}", filepath, why),
                Ok(file) => file,
            };
            match file.write_all(notes.as_bytes()) {
                Err(why) => panic!("Couldn't write to {:?}: {}", filepath, why),
                Ok(_) => println!("Wrote to {:?}", filepath),
            }
        }
        WriteModes::A => {
            println!("Appending to {:?}", filepath);
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn it_works() {
        // let result = add(2, 2);
        // assert_eq!(result, 4);
    }
}

/// A macro for printing code
#[macro_export]
macro_rules! execute {
    ($e:block) => {{
        println!("");
        println!("---");
        let code_str = stringify!($e);
        // remove brackets
        // code_str.
        for (i, line) in code_str
            .trim()
            .trim_start_matches("{")
            .trim_end_matches("}")
            .split(';')
            .enumerate()
        {
            let trimmed_line = line.trim();
            if !trimmed_line.is_empty() {
                println!("{}|    {}", i + 1, trimmed_line);
            }
        }

        let result = $e; // The block is still executed here
        println!("---");
        println!("   -> {:?}", result);
        println!("");

        result
    }};
    ($e:expr) => {
        // This takes the code and turns it into a string
        let code = stringify!($e);
        let result = $e;
        println!("");
        println!("{} ->", code);
        println!("---");
        println!("{}", result);
        println!("---");
        println!("");
    };

    // This macro arm matches a full function definition.
    (
        $(#[$outer:meta])* // Match any outer attributes like `#[allow(dead_code)]`
        $vis:vis fn $name:ident ($($arg:tt)*) $(-> $ret:ty)?
        $body:block
    ) => {
        $vis mod $name {
            $(#[$outer])*
            pub fn call($($arg)*) $(-> $ret)? {
                $body
            }
        pub const SOURCE_CODE: &'static str = stringify!(
                // We reconstruct the ideal signature for the documentation string
                $(#[$outer])*
                pub fn $name($($arg)*) $(-> $ret)?
                $body
            );
        }
    };

}
