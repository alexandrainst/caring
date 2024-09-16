pub mod element;
pub mod field;
pub mod math;
pub mod poly;
pub mod rayon;

#[allow(clippy::len_without_is_empty)]
pub trait Length {
    fn len(&self) -> usize;
}
