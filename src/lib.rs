pub mod encoding;
pub mod psuedocode;

use quick_xml::de::from_str;
use encoding::{EncodingIndex, InstructionSection};

use std::fs;
use std::path::Path;

static ENCODING_INDEX_PATH: &str = "encodingindex.xml"; 

pub struct SpecificationDirectory {
    dir_path: Box<Path>
}

impl SpecificationDirectory {
    pub fn new(dir_path: &str) -> Self {
        let path = Path::new(dir_path);
        Self {
            dir_path: path.into()
        }
    }
    
    pub fn get_encoding_index(&self) -> EncodingIndex {
        let index_path = self.dir_path.join(ENCODING_INDEX_PATH);
        read_encoding_index_from_file(index_path.to_str().expect("Invalid Path"))
    }

    pub fn get_instruction_section(&self, filepath: &str) -> InstructionSection {
        let instr_section_path = self.dir_path.join(filepath);
        read_instruction_section_from_file(instr_section_path.to_str().expect("Invalid Path"))
    }
}

pub fn read_instruction_section_from_buffer(buf: &str) -> InstructionSection {
    from_str(buf).expect("Failed to deserialize XML spec")
}

pub fn read_encoding_index_from_buffer(buf: &str) -> EncodingIndex {
    from_str(buf).expect("Failed to deserialize XML spec")
}

pub fn read_instruction_section_from_file(filepath: &str) -> InstructionSection {
    let buffer = fs::read_to_string(filepath)
        .expect("Failed to read file.");
    read_instruction_section_from_buffer(&buffer)
}

pub fn read_encoding_index_from_file(filepath: &str) -> EncodingIndex {
    let buffer = fs::read_to_string(filepath)
        .expect("Failed to read file.");
    read_encoding_index_from_buffer(&buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing() {
        
    }
}
