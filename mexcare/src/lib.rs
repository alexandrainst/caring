use rustmex::prelude::*;

#[rustmex::entrypoint]
fn hello_world(lhs: Lhs, rhs: Rhs) -> rustmex::Result<()> {
	println!("Hello Matlab!");
	Ok(())
}
