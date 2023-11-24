use std::convert::Infallible;

// TODO: Serde trait bounds on `T`
// TODO: Properly use this trait for other things (Connection/Agency etc.)
pub trait Channel {
    type Error;

    async fn send<T>(&self, msg: T) -> Result<(), Self::Error>;

    async fn recv<T>(&self) -> Result<T, Self::Error>;
}

// TODO: Handle other things than 1-of-2 OT.
// Just make everything 1-of-n?
pub trait ObliviousTransfer<C: Channel> {
    // TODO: Better and less overloaded names
    type Sender : ObliviousSend<C>;
    type Receiver : ObliviousReceive<C>; // Selector? Chooser?
}

// INFO: Consider if Channel should be a parameter or part of the object instance?
// Point for parameter: We always need a channel to send over (it is a transfer after all)
// Point against: It is less flexible and forces the implmentor to use a channe and forces the
// implmentor to use a channel
// Point against: It does allow us to remove the generic `C`
//
// INFO: Also, maybe we should join the ObliviousSend and ObliviousReceive traits?
// Does it make sense to keep them seperated, or should the ObliviousTransfer trait just have both
// implementations?
pub trait ObliviousSend<C: Channel> {
    type Error;

    async fn send<T>(pkg0: T, pkg1: T, channel: C) -> Result<(), Self::Error>;
}

pub trait ObliviousReceive<C: Channel> {
    type Error;

    async fn choose<T>(choice: bool, channel: C) -> Result<T, Self::Error>;
}




struct MockOT();

impl<C: Channel> ObliviousTransfer<C> for MockOT {
    type Sender = MockOTSender;

    type Receiver = MockOTReceiver;
}

struct MockOTSender();

impl<C: Channel> ObliviousSend<C> for MockOTSender {
    type Error = C::Error;

    async fn send<T>(pkg0: T, pkg1: T, channel: C) -> Result<(), Self::Error> {
        channel.send((pkg0, pkg1)).await?;
        Ok(())
    }
}


struct MockOTReceiver();

impl<C: Channel> ObliviousReceive<C> for MockOTReceiver {
    type Error = C::Error;

    async fn choose<T>(choice: bool, channel: C) -> Result<T, Self::Error> {
        let (pkg0, pkg1) = channel.recv().await?;
        Ok(if choice { pkg1 } else { pkg0 })
    }
}
