use plotters::prelude::*;

fn main() {
    let root_drawing_area = BitMapBackend::new("mesh.png", (600, 400)).into_drawing_area();

    root_drawing_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_drawing_area)
        .caption("Figure Sample", ("Arial", 30))
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(-10..30, 0..1_000)
        .unwrap();

    // draw mesh
    ctx.configure_mesh().draw().unwrap();

    // draw data
    ctx.draw_series(LineSeries::new((-10..=50).map(|x| (x, x * x)), &GREEN))
        .unwrap();

    ctx.draw_series(
        DATA_CIRCLES
            .iter()
            .map(|point| Circle::new(*point, 5, &RED)),
    )
    .unwrap();
}

const DATA_CIRCLES: [(i32, i32); 30] = [
    (-3, 1),
    (-2, 3),
    (4, 2),
    (3, 0),
    (6, -5),
    (3, 11),
    (6, 0),
    (2, 14),
    (3, 9),
    (14, 7),
    (8, 11),
    (10, 16),
    (7, 15),
    (13, 8),
    (17, 14),
    (13, 17),
    (19, 11),
    (18, 8),
    (15, 8),
    (23, 23),
    (15, 20),
    (22, 23),
    (22, 21),
    (21, 30),
    (19, 28),
    (22, 23),
    (30, 23),
    (26, 35),
    (33, 19),
    (26, 19),
];
