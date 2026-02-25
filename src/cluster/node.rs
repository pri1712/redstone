/// A node is an abstraction for a server, it bundles the name of the server and its IP address
/// into a single instance.
pub struct Node {
    //contains the mapping from ip address to dns name for a server
    pub address: String,
    pub dns_name: String,
}

impl Node {
    pub fn new(address: String, dns_name: String) -> Node {
        Self { address, dns_name }
    }
}