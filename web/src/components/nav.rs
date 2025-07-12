use dioxus::prelude::*;

/// A guide for the app
#[component]
pub fn NavBar() -> Element {
    rsx! {
        div { id: "navbar",
            "Nav"
        }
    }
}
