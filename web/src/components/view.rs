use crate::components::*;
use crate::Route;
use dioxus::prelude::*;

#[component]
pub fn OptimizersView() -> Element {
    rsx! {
        OptimizerDemo {}
    }
}

#[component]
pub fn PlaygroundView() -> Element {
    rsx! {
        MLPlayground {}
    }
}

#[component]
pub fn ArenaView() -> Element {
    rsx! {
        AlgorithmArenaSimple {}
    }
}

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
                            to: Route::ShowcaseView {},
                            "ML Library Showcase"
                        }
                    }
                    li {
                        Link {
                            active_class: "active",
                            class: "link_class",
                            to: Route::OptimizersView {},
                            "Optimizer Visualizer"
                        }
                    }
                    li {
                        Link {
                            active_class: "active",
                            class: "link_class",
                            to: Route::PlaygroundView {},
                            "🚀 ML Playground"
                        }
                    }
                    li {
                        Link {
                            active_class: "active",
                            class: "link_class",
                            to: Route::ArenaView {},
                            "🏁 ML Algorithm Arena"
                        }
                    }
                    li {
                        Link {
                            active_class: "active",
                            class: "link_class",
                            id: "link_id",
                            new_tab: false,
                            rel: "link_rel",
                            to: Route::CoursesView {},

                            "Coursera ML Course"
                        }
                    }
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
