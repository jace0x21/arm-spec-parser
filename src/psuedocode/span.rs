use core::ops::{Range, RangeTo, RangeFrom, RangeFull};
use std::str::{Chars, CharIndices, FromStr};

use nom::{
    AsBytes,
    Compare,
    CompareResult,
    Err,
    FindSubstring,
    FindToken,
    InputIter,
    InputLength,
    InputTake,
    InputTakeAtPosition,
    IResult,
    Offset,
    ParseTo,
    Slice,
};

use nom::error::{ErrorKind, ParseError};

#[derive(Clone)]
pub struct Span<'a> {
    pub input: &'a str,
    pub min_precedence: u32,
}

impl<'a> AsBytes for Span<'a> {
    fn as_bytes(&self) -> &[u8] {
        self.input.as_ref()
    }
}

impl<'a> Compare<&str> for Span<'a> {
  fn compare(&self, t: &str) -> CompareResult {
    self.as_bytes().compare(t.as_bytes())
  }

  fn compare_no_case(&self, t: &str) -> CompareResult {
    let pos = self.input
      .chars()
      .zip(t.chars())
      .position(|(a, b)| a.to_lowercase().ne(b.to_lowercase()));

    match pos {
      Some(_) => CompareResult::Error,
      None => {
        if self.input.len() >= t.len() {
          CompareResult::Ok
        } else {
          CompareResult::Incomplete
        }
      }
    }
  }
}

impl<'a> FindToken<u8> for Span<'a> {
    fn find_token(&self, token: u8) -> bool {
        self.as_bytes().find_token(token)
    }
}

impl<'a> FindSubstring<&str> for Span<'a> {
  fn find_substring(&self, substr: &str) -> Option<usize> {
    self.input.find(substr)
  }
}

impl<'a, R: FromStr> ParseTo<R> for Span<'a> {
  fn parse_to(&self) -> Option<R> {
    self.input.parse().ok()
  }
}

macro_rules! impl_slice_for { 
    ( $ty:ty ) => {
        impl<'a> Slice<$ty> for Span<'a> {
            fn slice(&self, range: $ty) -> Self {
                let slice = self.input.slice(range);
                return Self {
                    input: slice,
                    min_precedence: self.min_precedence,
                }
            }
        }
    };
}

impl_slice_for!(RangeFull);
impl_slice_for!(Range<usize>);
impl_slice_for!(RangeTo<usize>);
impl_slice_for!(RangeFrom<usize>);

impl<'a> Offset for Span<'a> {
    fn offset(&self, second: &Self) -> usize {
        let fst = self.input.as_ptr();
        let snd = second.input.as_ptr();

        snd as usize - fst as usize
    }
}

impl<'a> InputLength for Span<'a> {
    fn input_len(&self) -> usize {
        self.input.len()
    }
}

impl<'a> InputIter for Span<'a> {
    type Item = char;
    type Iter = CharIndices<'a>;
    type IterElem = Chars<'a>;
  
    fn iter_indices(&self) -> Self::Iter {
        self.input.char_indices()
    }
    
    fn iter_elements(&self) -> Self::IterElem {
        self.input.chars()
    }
    
    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool,
    {
        for (o, c) in self.input.char_indices() {
            if predicate(c) {
                return Some(o);
            }
        }
        None
    }

    fn slice_index(&self, count: usize) -> Result<usize, nom::Needed> {
        let mut cnt = 0;
        for (index, _) in self.input.char_indices() {
            if cnt == count {
                return Ok(index);
            }
            cnt += 1;
        }
        if cnt == count {
            return Ok(self.input.len());
        }
        Err(nom::Needed::Unknown)
    }
}

impl<'a> InputTake for Span<'a> {
    fn take(&self, count: usize) -> Self {
        self.slice(..count)
    }

    fn take_split(&self, count: usize) -> (Self, Self) {
        (self.slice(count..), self.slice(..count))
    }
}

impl<'a> InputTakeAtPosition for Span<'a> {
    type Item = char;

    fn split_at_position<P, E: ParseError<Self>>(&self, predicate: P) -> IResult<Self, Self, E>
    where
        P: Fn(Self::Item) -> bool,
    {
        match self.input.position(predicate) {
            Some(n) => Ok(self.take_split(n)),
            None => Err(Err::Incomplete(nom::Needed::new(1))),
        }
    }

    fn split_at_position1<P, E: ParseError<Self>>(
        &self,
        predicate: P,
        e: ErrorKind,
    ) -> IResult<Self, Self, E>
    where
        P: Fn(Self::Item) -> bool,
    {
        match self.input.position(predicate) {
            Some(0) => Err(Err::Error(E::from_error_kind(self.clone(), e))),
            Some(n) => Ok(self.take_split(n)),
            None => Err(Err::Incomplete(nom::Needed::new(1))),
        }
    }

    fn split_at_position_complete<P, E: ParseError<Self>>(
        &self,
        predicate: P,
    ) -> IResult<Self, Self, E>
    where
        P: Fn(Self::Item) -> bool,
    {
        match self.split_at_position(predicate) {
            Err(Err::Incomplete(_)) => Ok(self.take_split(self.input.input_len())),
            res => res,
        }
    }

    fn split_at_position1_complete<P, E: ParseError<Self>>(
        &self,
        predicate: P,
        e: ErrorKind,
    ) -> IResult<Self, Self, E>
    where
        P: Fn(Self::Item) -> bool,
    {
        match self.input.position(predicate) {
            Some(0) => Err(Err::Error(E::from_error_kind(self.clone(), e))),
            Some(n) => Ok(self.take_split(n)),
            None => {
                if self.input.input_len() == 0 {
                    Err(Err::Error(E::from_error_kind(self.clone(), e)))
                } else {
                    Ok(self.take_split(self.input.input_len()))
                }
            }
        } 
    }
}

