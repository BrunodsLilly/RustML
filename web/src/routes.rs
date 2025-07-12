use crate::components::*;
use dioxus::prelude::*;

#[derive(Routable, Clone, PartialEq)]
pub enum Route {
    #[route("/")]
    MainView,
    #[route("/courses")]
    CoursesView,
}
