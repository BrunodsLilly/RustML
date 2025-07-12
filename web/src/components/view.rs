use crate::components::*;
use crate::routes::Route;
use dioxus::prelude::*;

#[component]
pub fn MainView() -> Element {
    rsx! {
        div { id: "main",
            NavBar {},
            div { id: "title",
                h1 { "Bruno" }
            }
            div { id: "app-list",
                ul {
                    li {
                        Link {
                            active_class: "active",
                            class: "link_class",
                            id: "link_id",
                            new_tab: false,
                            rel: "link_rel",
                            to: Route::CoursesView {},

                            "A fully configured link"
                        }
                    }
                    li { a { href:"/", "app 2" } }
                }
            }
        }
    }
}

#[component]
pub fn CoursesView() -> Element {
    rsx! {
        "courses"
        ul {
            li {
                Link {
                    to: Route::MainView {},
                    "Main"
                }
            }
        }
    }
}
