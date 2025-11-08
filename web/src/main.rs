mod components;

use crate::components::*;

use dioxus::prelude::*;

static CSS: Asset = asset!("/assets/main.css");

#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    MainView,
    #[route("/courses")]
    CoursesView,
    #[route("/showcase")]
    ShowcaseView,
    #[route("/optimizers")]
    OptimizersView,
    #[route("/playground")]
    PlaygroundView,
}

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    rsx! {
        document::Stylesheet { href: CSS }
        Router::<Route> {}
    }
}
