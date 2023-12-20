pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Point2<T> {
    pub fn new(x:T, y:T) -> Self {
        Self {x, y}
    }
}

impl<T> Point3<T> {
    pub fn new(x:T, y:T, z:T) -> Self {
        Self {x, y, z}
    }
}

pub trait Type<T> {
    type Point;
}

pub struct TwoDim;
pub struct ThreeDim;

impl<T> Type<T> for TwoDim {
    type Point = Point2<T>;
}

impl<T> Type<T> for ThreeDim {
    type Point = Point3<T>;
}

pub type Point<T, D> = <D as Type<T>>::Point;