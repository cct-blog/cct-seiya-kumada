mod point;
use point::{Point, TwoDim, ThreeDim};

fn main() {
    let _x2 = Point::<i32, TwoDim>::new(3, 3);
    let _x3 = Point::<i32, ThreeDim>::new(3, 3,3);
}
