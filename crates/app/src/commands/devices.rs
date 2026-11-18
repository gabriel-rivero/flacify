//! `flacify devices`: the capture and playback devices this machine offers.

use anyhow::Result;

use crate::device;

pub(crate) fn run() -> Result<()> {
    let list = device::list()?;
    print!("{}", device::describe(&list));
    // Names are what every device flag takes, and the two demands below are the ones that
    // turn a name into a refusal, so say them next to the list.
    println!("\nsegment playback and capture need a 48 kHz stereo device");
    Ok(())
}
