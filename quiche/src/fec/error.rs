/// Various FEC related error types
#[derive(Debug)]
pub enum FecError {
    /// Error from moeprlnc
    MoepInitError,

    /// Indicates that sliding window is empty
    SlidingWindowIsEmpty,

    /// Indicates that payload is too long
    PayloadLongerThanMaximum,

    /// Indicates that payload is too short
    RepairSymbolLen,
}
