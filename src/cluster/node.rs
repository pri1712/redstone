/// A node is an abstraction for a server, it bundles the name of the server and its IP address
/// into a single instance.
pub struct Node {
    //contains the mapping from ip address to dns name for a server
    pub address: String,
    pub name: String,
}

impl Node {
    pub fn new(address: impl Into<String>, name: impl Into<String>) -> Node {
        Self {
            address: address.into(),
            name: name.into(),
        }
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new() {
        let node = Node::new("127.0.0.1", "localhost");
        assert_eq!(node.address, String::from("127.0.0.1"));
    }
}