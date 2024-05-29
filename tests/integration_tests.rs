use std::fs;

use arm_spec_parser;

#[test]
fn test_adr() {
    let xml_data = fs::read_to_string("./tests/adr.xml")
        .expect("Unable to read adr.xml test file");
    let deserialized_xml = arm_spec_parser::read_instruction_section_from_buffer(&xml_data);
}

#[test]
fn test_abs() {
    let xml_data = fs::read_to_string("./tests/abs.xml")
        .expect("Unable to read adr.xml test file");
    let deserialized_xml = arm_spec_parser::read_instruction_section_from_buffer(&xml_data);
}

#[test]
fn test_at_sys() { 
    let xml_data = fs::read_to_string("./tests/at_sys.xml")
        .expect("Unable to read adr.xml test file");
    let deserialized_xml = arm_spec_parser::read_instruction_section_from_buffer(&xml_data);
}

#[test]
fn test_encoding_index() {
    let xml_data = fs::read_to_string("./tests/encodingindex.xml")
        .expect("Unable to read adr.xml test file");
    let deserialized_xml = arm_spec_parser::read_encoding_index_from_buffer(&xml_data);
    assert_eq!(deserialized_xml.iclass_map.keys().collect::<Vec<&String>>().len(), 129);
}
