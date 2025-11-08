use notes::write_note;

fn main() {
    let filepath = "/tmp/del.md";
    let mode = notes::WriteModes::W;
    let notes = "Bruno's notes";

    write_note(filepath, notes, mode);
}
