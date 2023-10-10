//! This a experiment for a structure to run MPC programs


use tokio::io::{AsyncRead, AsyncWrite};


use crate::connection::Network;


pub struct Engine<R : AsyncRead + Unpin, W : AsyncWrite + Unpin> {
    // Should probably just be a map with ID's to parties.
    // However we still need a good way to get IDs.
    // One way could be to do as Fresco does and just supply them in the beginning.
    // I, however, find that wildly annoying when testing.
    // Sure, in the real world we would know who we talk with before hand,
    // but then we would also have certificates and other stuff.
    //
    // A quick and dirty solution to ID resolution would just let everyone
    // pick a random number and then sort and enumerate them.
    // Then we just need a broadcast to ensure that everyone have the same IDs.
    // (Probably also sending the party with ID 'i' a message that we know they are 'i')
    //
    // If everything goes well we should then have some IDs, however they wouldn't be 'fair' IDs,
    // in the sense that a given party can always choose a low random value.
    // So don't depend anything on being number one!
    network: Network<R,W>,
}



impl<R : AsyncRead + Unpin, W : AsyncWrite + Unpin> Engine<R,W> {


}

