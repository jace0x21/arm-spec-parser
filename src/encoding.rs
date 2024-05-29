use std::collections::HashMap;
use std::fmt;

use serde::{de::{Error, MapAccess, Visitor}, Deserialize, Deserializer};

pub struct EncodingIndex {
    pub hierarchy: Hierarchy,
    pub iclass_map: HashMap<String, Vec<IClassSect>>,
}

struct EncodingIndexVisitor;

impl<'de> Visitor<'de> for EncodingIndexVisitor {
    type Value = EncodingIndex;

    // Format a message stating what data this Visitor expects to receive.
    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("EncodingIndex")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        #[derive(Debug, Deserialize)]
        struct Ignore {}

        let _entry = access.next_entry::<String, String>();
        match access.next_key::<&str>() {
            Ok(Some("hierarchy")) => (),
            Ok(Some(&_)) => return Err(Error::custom("hierarchy not found as first child")),
            Ok(None) => return Err(Error::custom("Failed to read key")),
            Err(e) => return Err(e),
        }
        
        let hierarchy = match access.next_value::<Hierarchy>() {
            Ok(value) => value,
            Err(_e) => return Err(Error::custom("Failed to deserialize hierarchy")),
        };

        let mut iclass_map: HashMap<String, Vec<IClassSect>> = HashMap::new();
        let mut current_id: String = String::new();
        let mut current_vec: Vec<IClassSect> = Vec::new();
        while let Ok(Some(current_key)) = access.next_key::<String>() {
            match current_key.as_str() {
                "funcgroupheader" =>
                    if current_vec.is_empty() {
                        current_id = access.next_value::<FunctionalGroupHeader>()?.id
                    } else {
                        assert!(!current_id.is_empty());
                        iclass_map.insert(current_id, current_vec);
                        current_vec = Vec::new();
                        current_id = access.next_value::<FunctionalGroupHeader>()?.id
                    }
                "iclass_sect" => current_vec.push(access.next_value::<IClassSect>()?),
                &_ => { let _ = access.next_value::<Ignore>(); }
            }
        }
        if !current_vec.is_empty() {
            iclass_map.insert(current_id, current_vec);
        }
        
        Ok(EncodingIndex {
            hierarchy,
            iclass_map,
        })
    }
}

impl<'de> Deserialize<'de> for EncodingIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
             D: Deserializer<'de> {
        deserializer.deserialize_struct("", &[], EncodingIndexVisitor)
    }
}

#[derive(Deserialize)]
pub struct FunctionalGroupHeader {
    #[serde(rename = "@id")]
    pub id: String,
}

#[derive(Deserialize)]
pub struct Hierarchy {
    pub regdiagram: RegDiagram,
    #[serde(rename = "node")]
    pub node_tree: Vec<Node>,
}

#[derive(Deserialize)]
pub struct IClassSect {
    #[serde(rename = "@id")]
    pub name: String,
    pub regdiagram: RegDiagram,
    pub instructiontable: InstructionTable,
}

#[derive(Deserialize)]
pub struct InstructionTable {
    pub thead: TableHeader,
    pub tbody: TableBody,
}

#[derive(Deserialize)]
pub struct TableHeader {
    #[serde(rename = "tr")]
    pub rows: Vec<HeaderRow>,
}

#[derive(Deserialize)]
pub struct TableBody {
    #[serde(rename = "tr")]
    pub rows: Vec<BodyRow>,
}

#[derive(Deserialize)]
pub struct HeaderRow {
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(rename = "th")]
    #[serde(default)]
    pub items: Vec<Item>
}

#[derive(Deserialize)]
pub struct BodyRow {
    #[serde(rename = "@iformfile")]
    pub iformfile: Option<String>,
    #[serde(rename = "@label")]
    pub label: Option<String>,
    #[serde(rename = "@encname")]
    pub encname: String,
    #[serde(rename = "td")]
    pub items: Vec<Item>
}

#[derive(Deserialize)]
pub struct Item {
    #[serde(rename = "@class")]
    pub class: String,
    #[serde(rename = "$text")]
    pub value: Option<String>,
}

#[derive(Clone, Deserialize)]
struct RawNode {
    #[serde(rename = "@groupname")]
    group_name: Option<String>,
    #[serde(rename = "@iclass")]
    iclass_name: Option<String>,
    decode: Decode,
    regdiagram: Option<RegDiagram>,
    #[serde(rename = "node")]
    #[serde(default)]
    children: Vec<RawNode>,
}

#[derive(Clone, Deserialize)]
#[serde(from = "RawNode")]
pub enum Node {
    Class(ClassNode),
    Group(GroupNode),
}

// impl Node {
//     fn get_name() -> String {
//         match self {
//             Some(GroupNode) => node.group_name,
//             Some(ClassNode) => 
//         }
//     }
// }

impl From<RawNode> for Node {
    fn from(node: RawNode) -> Self {
        let children = node.children
            .iter()
            .map(|x| Node::from(x.clone()))
            .collect();
        if let Some(name) = node.group_name {
            return Node::Group(GroupNode { 
                name: name.to_string(),
                decode: node.decode.clone(),
                regdiagram: node.regdiagram
                    .expect("Error: Group Node with no regdiagram")
                    .clone(),
                children,
            })
        } else if let Some(name) = node.iclass_name {
            return Node::Class(ClassNode {
                name: name.to_string(),
                decode: node.decode.clone(),
            })
        } else {
            panic!("Unimplemented Node type!");
        }
    }
}

#[derive(Clone, Deserialize)]
pub struct GroupNode {
    #[serde(rename = "@groupname")]
    pub name: String,
    pub regdiagram: RegDiagram,
    pub decode: Decode,
    pub children: Vec<Node>,
}

#[derive(Clone, Deserialize)]
pub struct ClassNode {
    #[serde(rename = "@iclass")]
    pub name: String,
    pub decode: Decode,
}

#[derive(Clone, Deserialize)]
pub struct Decode {
    #[serde(rename = "box")]
    pub boxes: Vec<EncodingBox>,
}

#[derive(Deserialize)]
pub struct InstructionSection {
    #[serde(rename = "@id")]
    pub id: String,
    pub docvars: DocVars,
    pub alias_list: Option<AliasList>,
    pub classes: ClassContainer,
    pub explanations: Explanations,
}

impl InstructionSection {
    pub fn get_regdiagrams(&self) -> HashMap<String, Vec<EncodingBox>> {
        self.classes.iclass_list
            .iter()
            .map(|class| (class.name.clone(), class.regdiagram.boxes.clone()))
            .collect()
    }

    pub fn get_encodings(&self) -> Vec<Encoding> {
        self.classes.iclass_list
            .iter()
            .map(|class| class.encodings.clone())
            .flatten()
            .collect()
    }

    pub fn get_docvars(&self) -> HashMap<&str, &str> {
        self.docvars.get_map()
    }
}

#[derive(Clone, Deserialize)]
pub struct DocVars {
    #[serde(rename = "docvar")]
    #[serde(default)]
    vars: Vec<DocVar>
}

impl DocVars {
    pub fn get_map(&self) -> HashMap<&str, &str> {
        self.vars
            .iter()
            .map(|x| (x.key.as_str(), x.value.as_str()))
            .collect()
    }
}

#[derive(Clone, Deserialize)]
pub struct DocVar {
    #[serde(rename = "@key")]
    key: String,
    #[serde(rename = "@value")]
    value: String,
}

#[derive(Deserialize)]
pub struct AliasList {
    #[serde(rename = "aliasref")]
    #[serde(default)]
    pub aliasrefs: Vec<AliasRef>,
}

#[derive(Deserialize)]
pub struct AliasRef { 
    #[serde(rename = "@aliaspageid")]
    pub aliaspageid: String,
    #[serde(rename = "@aliasfile")]
    pub aliasfile: String,
    #[serde(rename = "aliaspref")]
    pub aliasprefs: Vec<AliasPref>,
}

#[derive(Deserialize)]
pub struct AliasPref {
    #[serde(rename = "$value")]
    pub value: Vec<AliasPrefValue>,
}

#[derive(Deserialize)]
pub enum AliasPrefValue {
    #[serde(rename = "a")]
    Link(LinkedText),
    #[serde(rename = "$text")]
    Text(String),
}

#[derive(Deserialize)]
pub struct ClassContainer {
    #[serde(rename = "iclass")]
    pub iclass_list: Vec<EncodingClass>
}

#[derive(Deserialize)]
pub struct Explanations {
    #[serde(rename = "explanation")]
    #[serde(default)]
    pub contents: Vec<Explanation>,
}

#[derive(Deserialize)]
pub struct Explanation {
    #[serde(rename = "@enclist")]
    pub enclist: String,
    pub symbol: Symbol,
    #[serde(alias = "definition")]
    #[serde(alias = "account")]
    pub detail: SymbolDetail,
}

#[derive(Deserialize)]
pub struct Symbol {
    #[serde(rename = "@link")]
    pub link: String,
    #[serde(rename = "$value")]
    pub value: String
}

#[derive(Deserialize)]
pub struct SymbolDetail {
    #[serde(rename = "@encodedin")]
    pub encodedin: String,
    pub table: Option<ValueTable>,
}

#[derive(Deserialize)]    
pub struct ValueTable {
    pub tgroup: ValueTableGroup,
}

#[derive(Deserialize)]    
pub struct ValueTableGroup {
    pub thead: ValueTableSection,
    pub tbody: ValueTableSection,
}

#[derive(Deserialize)]    
pub struct ValueTableSection {
    #[serde(rename = "row")]
    pub rows: Vec<ValueTableRow>,
}

#[derive(Deserialize)]    
pub struct ValueTableRow {
    #[serde(rename = "entry")]
    pub entries: Vec<ValueTableEntry>,
}

#[derive(Deserialize)]    
pub struct ValueTableEntry {
    #[serde(rename = "@class")]
    pub class: String,
    #[serde(rename = "$value")]
    pub value: Option<EntryValue>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryValue {
    ArchVariants(ArchVariantList),
    #[serde(rename = "$text")]
    Text(String),
    Para(Paragraph),
}

#[derive(Deserialize)]
pub struct Paragraph { }

#[derive(Deserialize)]
pub struct ArchVariantList { }

#[derive(Deserialize)]
pub struct EncodingClass {
    #[serde(rename = "@name")]
    pub name: String,
    pub regdiagram: RegDiagram,
    #[serde(rename = "encoding")]
    pub encodings: Vec<Encoding>,
}

#[derive(Clone, Deserialize)]
pub struct RegDiagram {
    #[serde(rename = "@form")]
    pub form: String,
    #[serde(rename = "@psname")]
    pub psname: Option<String>,
    #[serde(rename = "box")]
    pub boxes: Vec<EncodingBox>,
}

#[derive(Clone, Deserialize)]
pub struct Encoding {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@label")]
    pub label: Option<String>,
    pub docvars: DocVars,
    #[serde(rename = "box")]
    #[serde(default)]
    pub boxes: Vec<EncodingBox>,
    pub asmtemplate: Option<AsmTemplate>,
}

#[derive(Clone, Deserialize)]
pub struct AsmTemplate {
    #[serde(rename = "$value")]
    pub text: Vec<Token>,
}

#[derive(Clone, Deserialize)]
pub enum Token {
    #[serde(rename = "text")]
    Plain(Text),
    #[serde(rename = "a")]
    Link(LinkedText),
}

#[derive(Clone, Deserialize)]
pub struct Text {
    #[serde(rename = "$value")]
    #[serde(default)]
    pub value: String
}

#[derive(Clone, Deserialize)]
pub struct LinkedText {
    #[serde(rename = "@link")]
    pub link: String,
    #[serde(rename = "$value")]
    #[serde(default)]
    pub value: String,
}

impl Encoding {
    pub fn get_mnemonic(&self) -> String {
        self.docvars.get_map()["mnemonic"].to_string()
    }
}

#[derive(Clone, Deserialize)]
pub struct EncodingBox {
    #[serde(rename = "@hibit")]
    pub hibit: usize,
    #[serde(rename = "@width")]
    pub width: Option<usize>,
    #[serde(rename = "@settings")]
    pub settings: Option<u32>,
    #[serde(rename = "@name")]
    pub name: Option<String>,
    #[serde(rename = "@psbits")]
    pub psbits: Option<String>,
    #[serde(rename = "c")]
    pub bit_pattern: Vec<Bits>,
}

impl EncodingBox {
    pub fn get_bits(&self) -> Vec<&Bit> {
        self.bit_pattern.iter()
            .map(|x| &x.value)
            .flatten()
            .collect()
    }
}

#[derive(Clone, Deserialize)]
#[serde(from = "RawBits")]
pub struct Bits {
    pub count: usize,
    pub value: Vec<Bit>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Bit {
    Concrete(u8),
    Not(u8),
    Unused(u8),
    Wildcard,
}

// #[derive(Clone, Debug, Eq, PartialEq)]
// struct BitSequence {
//     bits: Vec<Bit>,
//     ne: bool,
// }

impl From<RawBits> for Bits {
    fn from(raw: RawBits) -> Bits {
        let bit_vec = match raw.value {
            Some(bit_string) => to_bit_vec(bit_string),
            None => vec![Bit::Wildcard; raw.count],
        };
        Bits {
            count: raw.count,
            value: bit_vec,
        }
    }
}

pub fn to_bit_vec(bit_string: String) -> Vec<Bit> {
    let to_u8 = |c: char| -> u8 {
        match c {
            '0' => 0,
            '1' => 1,
            _ => panic!("Unexpected bit while parsing: {}", c),
        }
    };
    let mut bits: Vec<Bit> = Vec::new();
    let mut ne = false;
    let mut chars = bit_string.chars();
    while let Some(c) = chars.next() {
        match c {
            '0' | '1' => if ne { bits.push(Bit::Not(to_u8(c))) } else { bits.push(Bit::Concrete(to_u8(c))); }
            'x' => bits.push(Bit::Wildcard),
            'N' => bits.push(Bit::Not(1)),
            'Z' => bits.push(Bit::Not(0)),
            '!' =>  {
                ne = true;
                chars.next();
                chars.next();
            } 
            '(' => if let Some(value) = chars.next() {
                bits.push(Bit::Unused(to_u8(value)));
                chars.next();
            }
            _ => panic!("Error: failed parsing bit sequence: {}", bit_string),
        }
    }
    return bits
}

#[derive(Deserialize)]
struct RawBits {
    #[serde(rename = "@colspan")]
    #[serde(default = "RawBits::default_count")]
    count: usize,
    #[serde(rename = "$text")]
    value: Option<String>,
}

impl RawBits {
    fn default_count() -> usize { 1 }
}

mod tests {
    use super::*;
    use quick_xml::de::from_str;

    #[test]
    fn test_bits() {
        let bits_xml = "<c>0</c>";
        let bits: Bits = from_str(&bits_xml).expect("Deserialization failed");
        assert_eq!(bits.count, 1);
        assert_eq!(bits.value, vec![
            Bit::Concrete(0),
        ]);

        let bits_xml = "<c>1</c>";
        let bits: Bits = from_str(&bits_xml).expect("Deserialization failed");
        assert_eq!(bits.count, 1);
        assert_eq!(bits.value, vec![
            Bit::Concrete(1),
        ]);
    }

    #[test]
    fn test_bits_sequence() {
        let bits_xml = "<c colspan=\"3\">000</c>";
        let bits: Bits = from_str(&bits_xml).expect("Deserialization failed");
        assert_eq!(bits.count, 3);
        assert_eq!(bits.value, vec![
            Bit::Concrete(0),
            Bit::Concrete(0),
            Bit::Concrete(0),
        ]);
    }

    #[test]
    fn test_wildcard_bits() {
        let bits_xml = "<c colspan=\"3\">0x01</c>";
        let bits: Bits = from_str(&bits_xml).expect("Deserialization failed");
        assert_eq!(bits.count, 3);
        assert_eq!(bits.value, vec![
            Bit::Concrete(0),
            Bit::Wildcard,
            Bit::Concrete(0),
            Bit::Concrete(1),
        ]);
    }

    #[test]
    fn test_empty_bits() {
        let bits_xml = "<c colspan=\"3\"></c>";
        let bits: Bits = from_str(&bits_xml).expect("Deserialization failed");
        assert_eq!(bits.count, 3);
        assert_eq!(bits.value, vec![
            Bit::Wildcard,
            Bit::Wildcard,
            Bit::Wildcard,
        ]);
    }

    #[test]
    fn test_ne_bits() {
        let bits_xml = "<c colspan=\"3\">!= 000</c>";
        let bits: Bits = from_str(&bits_xml).expect("Deserialization failed");
        assert_eq!(bits.count, 3);
        assert_eq!(bits.value, vec![
            Bit::Not(0),
            Bit::Not(0),
            Bit::Not(0),
        ]);
    }

    #[test]
    fn test_multiple_bit_sequences() {
        let bits_xml = "<c colspan=\"3\">1!= 010</c>";
        let bits: Bits = from_str(&bits_xml).expect("Deserialization failed");
        assert_eq!(bits.count, 3);
        assert_eq!(bits.value, vec![
            Bit::Concrete(1),
            Bit::Not(0),
            Bit::Not(1),
            Bit::Not(0),
        ]);
    }

    #[test]
    fn test_encoding_box() {
        let xml = "
            <box hibit=\"31\" name=\"op\" usename=\"1\" settings=\"1\" psbits=\"x\">
                <c>0</c>
            </box>";
        let ebox: EncodingBox = from_str(&xml).expect("Deserialization failed");
        assert_eq!(ebox.hibit, 31);
        assert_eq!(ebox.name, Some(String::from("op")));
    }

    #[test]
    fn test_instruction_section() {
        let xml = "
            <instructionsection id=\"ADR\" title=\"ADR -- A64\" type=\"instruction\">
            <docvars>
                <docvar key=\"address-form\" value=\"literal\"/>
                <docvar key=\"instr-class\" value=\"general\"/>
                <docvar key=\"isa\" value=\"A64\"/>
                <docvar key=\"mnemonic\" value=\"ADR\"/>
                <docvar key=\"offset-type\" value=\"off19s\"/>
            </docvars>
            <alias_list howmany=\"0\"/>
            <classes>
            <iclass name=\"Literal\" oneof=\"1\" id=\"iclass_literal\" no_encodings=\"1\" isa=\"A64\">
                <regdiagram form=\"32\" psname=\"A64.dpimm.pcreladdr.ADR_only_pcreladdr\" tworows=\"1\">
                <box hibit=\"31\" name=\"op\" usename=\"1\" settings=\"1\" psbits=\"x\">
                    <c>0</c>
                </box>
                <box hibit=\"30\" width=\"2\" name=\"immlo\" usename=\"1\">
                    <c colspan=\"2\"/>
                </box>
                <box hibit=\"28\" width=\"5\" settings=\"5\">
                    <c>1</c>
                    <c>0</c>
                    <c>0</c>
                    <c>0</c>
                    <c>0</c>
                </box>
                <box hibit=\"23\" width=\"19\" name=\"immhi\" usename=\"1\">
                    <c colspan=\"19\"/>
                </box>
                <box hibit=\"4\" width=\"5\" name=\"Rd\" usename=\"1\">
                    <c colspan=\"5\"/>
                </box>
                </regdiagram>
                <encoding name=\"ADR_only_pcreladdr\" oneofinclass=\"1\" oneof=\"1\" label=\"\">
                    <docvars>
                        <docvar key=\"isa\" value=\"A64\"/>
                        <docvar key=\"mnemonic\" value=\"ADR\"/>
                        <docvar key=\"address-form\" value=\"literal\"/>
                        <docvar key=\"instr-class\" value=\"general\"/>
                        <docvar key=\"offset-type\" value=\"off19s\"/>
                    </docvars>
                </encoding>
            </iclass>
            </classes>
            <explanations scope=\"all\">
            <explanation enclist=\"ADR_only_pcreladdr\" symboldefcount=\"1\">
            <symbol link=\"XdOrXZR__6\">&lt;Xd&gt;</symbol>
            <account encodedin=\"Rd\">
            <intro>
                <para>Is the 64-bit name of the general-purpose destination register, encoded in the \"Rd\" field.</para>
            </intro>
            </account>
            </explanation>
            <explanation enclist=\"ADR_only_pcreladdr\" symboldefcount=\"1\">
            <symbol link=\"immhiimmlo_offset\">&lt;label&gt;</symbol>
            <account encodedin=\"immhi:immlo\">
            <intro>
                <para>Is the program label whose address is to be calculated. Its offset from the address of this instruction, in the range +/-1MB, is encoded in \"immhi:immlo\".</para>
            </intro>
            </account>
            </explanation>
            </explanations>
            </instructionsection>";
        let instr_section: InstructionSection = from_str(&xml).expect("Deserialization failed");
        let encodings = instr_section.get_regdiagrams();
        assert_eq!(encodings.keys().len(), 1);
        assert!(encodings.contains_key("Literal"));
        assert_eq!(encodings["Literal"][0].hibit, 31);
        assert_eq!(encodings["Literal"][0].name, Some(String::from("op")));
    }
}
