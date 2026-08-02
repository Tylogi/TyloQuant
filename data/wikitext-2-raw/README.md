# WikiText-2 raw test split

`wiki.test.raw` is the ordered `test` split of the
`wikitext-2-raw-v1` configuration from
[Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext).
MFQ uses this corpus for reproducible perplexity, KLD, and same-top
evaluation. The file is stored byte-for-byte so Git does not change line
endings and therefore cannot change tokenization.

- SHA-256: `4f3e427e4aff23acff917a8a51aeae21bf12832256e26914c8fc250dcef4103c`
- Upstream licenses: CC BY-SA 3.0 and GFDL
- Upstream paper: Stephen Merity, Caiming Xiong, James Bradbury, and
  Richard Socher, *Pointer Sentinel Mixture Models*, 2016.

The dataset remains subject to its upstream licenses; MFQ's Apache-2.0
license does not replace them.
