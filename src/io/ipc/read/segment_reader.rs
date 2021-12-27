use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use arrow_format::ipc;
use arrow_format::ipc::flatbuffers::VerifierOptions;

use crate::error::{ArrowError, Result};
use crate::io::ipc::read::reader::read_dictionary_message;
use crate::io::ipc::read::{read_dictionary, FileMetadata};
use crate::io::ipc::{convert, ARROW_MAGIC};

/// Read the IPC file's metadata
pub fn read_file_segment_metadata<R: Read + Seek>(
    reader: &mut R,
    start: u64,
    end: u64,
) -> Result<FileMetadata> {
    // check if header and footer contain correct magic bytes
    let mut magic_buffer: [u8; 6] = [0; 6];
    reader.seek(SeekFrom::Start(start))?;
    reader.read_exact(&mut magic_buffer)?;
    if magic_buffer != ARROW_MAGIC {
        return Err(ArrowError::OutOfSpec(format!(
            "Arrow file segment [{}, {}] does not contain correct header",
            start, end
        )));
    }
    reader.seek(SeekFrom::Start(end - 6))?;
    reader.read_exact(&mut magic_buffer)?;
    if magic_buffer != ARROW_MAGIC {
        return Err(ArrowError::OutOfSpec(format!(
            "Arrow file segment [{}, {}] does not contain correct footer",
            start, end
        )));
    }
    // read footer length
    let mut footer_size: [u8; 4] = [0; 4];
    reader.seek(SeekFrom::Start(end - 10))?;
    reader.read_exact(&mut footer_size)?;
    let footer_len = i32::from_le_bytes(footer_size);

    // read footer
    let mut footer_data = vec![0; footer_len as usize];
    reader.seek(SeekFrom::Start(end - 10 - footer_len as u64))?;
    reader.read_exact(&mut footer_data)?;

    // set flatbuffer verification options to the same settings as the C++ arrow implementation.
    // Heuristic: tables in a Arrow flatbuffers buffer must take at least 1 bit
    // each in average (ARROW-11559).
    // Especially, the only recursive table (the `Field` table in Schema.fbs)
    // must have a non-empty `type` member.
    let verifier_options = VerifierOptions {
        max_depth: 128,
        max_tables: footer_len as usize * 8,
        ..Default::default()
    };
    let footer = ipc::File::root_as_footer_with_opts(&verifier_options, &footer_data[..]).map_err(
        |err| {
            ArrowError::OutOfSpec(format!(
                "Unable to get root as footer in segment [{}, {}]: {:?}",
                start, end, err
            ))
        },
    )?;

    let mut blocks = footer
        .recordBatches()
        .ok_or_else(|| {
            ArrowError::OutOfSpec(format!(
                "Unable to get record batches from footer in segment [{}, {}]",
                start, end
            ))
        })?
        .to_vec();

    blocks
        .iter_mut()
        .for_each(|b| b.set_offset(b.offset() + start as i64));

    let ipc_schema = footer.schema().ok_or_else(|| {
        ArrowError::OutOfSpec(format!(
            "Unable to get the schema from footer in segment [{}, {}]",
            start, end
        ))
    })?;
    let (schema, is_little_endian) = convert::fb_to_schema(ipc_schema);
    let schema = Arc::new(schema);

    let mut dictionaries = Default::default();

    let dictionary_blocks = footer.dictionaries().ok_or_else(|| {
        ArrowError::OutOfSpec(format!(
            "Unable to get dictionaries from footer in segment [{}, {}]",
            start, end
        ))
    })?;

    let mut data = vec![];
    for block in dictionary_blocks {
        let offset = block.offset() as u64 + start;
        let length = block.metaDataLength() as u64;
        read_dictionary_message(reader, offset, &mut data)?;

        let message = ipc::Message::root_as_message(&data).map_err(|err| {
            ArrowError::OutOfSpec(format!(
                "Unable to get root as message in segment [{}, {}]: {:?}",
                start, end, err
            ))
        })?;

        match message.header_type() {
            ipc::Message::MessageHeader::DictionaryBatch => {
                let block_offset = offset + length;
                let batch = message.header_as_dictionary_batch().unwrap();
                read_dictionary(
                    batch,
                    &schema,
                    is_little_endian,
                    &mut dictionaries,
                    reader,
                    block_offset,
                )?;
            }
            t => {
                return Err(ArrowError::OutOfSpec(format!(
                    "Expecting DictionaryBatch in dictionary blocks, found {:?}.",
                    t
                )));
            }
        };
    }
    Ok(FileMetadata {
        schema,
        is_little_endian,
        blocks,
        dictionaries,
        version: footer.version(),
    })
}
