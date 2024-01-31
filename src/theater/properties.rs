/// Trait for marking protocol/schemes as having specific security properties.
///
/// The idea is marking protocols with these properties.
///
/// It would be really cool if some of these could be autotraits in some way,
/// such that we can say a protocol is malicious secure if its subprotocols are malicious secure.
///

/// Marker trait for a passive secure protocol
///
/// Requirements for passive secure protocols are just that the parties
/// must adhere to the protocol.
trait PassiveSecure {}

/// Marker trait for a covert secure protocol
///
/// Requirements for passive covert protocols are just that the parties
/// must adhere to the protocol.
/// If they don't adhere to the protocol they should be caught with a properbility.
trait CovertSecure: PassiveSecure {}
impl<T: CovertSecure> PassiveSecure for T {}

/// Marker trait for a malicious secure protocol
///
/// Malicious secure protocols are secure against active malicious adversary.
trait MaliciousSecure: CovertSecure {}
impl<T: MaliciousSecure> CovertSecure for T {}

// Marker trait for protocols which hold up against quantum adversaries (?)
trait QuantumSecure {}
