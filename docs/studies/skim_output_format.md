# Skim output format study (2026-10-04)

Moved from docs/skimming.md on 2026-10-11; the conclusions are the defaults of the
skim campaign (`zstd:19`, 2 GB parts, packed chunks).


Memory: a boolean-masked awkward chunk keeps the whole chunk's buffers (all
branches) alive, so buffering masked chunks until the writer's flush grew to
several GB per worker at low pass rates (the first LPC test job was held at
4 GB). Since 2026-10-05 the selected events are packed (`ak.to_packed`)
before buffering: peak memory for a 190k-event WW file went from 2.7 GB to
0.65 GB with no growth across inputs.

Measured on one million 2024 ttbar events with the 856 kept branches, local
disk, one core; "read" is the analyzer-style read of the 112 columns the
analysis touches in 250k-event chunks (about 37 % of the bytes).

| Compression | bytes/event | write s per 1M events | read, k events/s |
| --- | --- | --- | --- |
| lz4:4 | 4700 | 40 | 857 |
| zlib:1 (old skims) | 3077 | 86 | 216 |
| zstd:3 | 3042 | 50 | 335 |
| zlib:6 | 2962 | 474 | 313 |
| zstd:9 | 2875 | 102 | 383 |
| zstd:15 | 2804 | 492 | 421 |
| **zstd:19** | **2304** | 1029 | 341 |
| lzma:5 | 1918 | 1119 | 34 |
| lzma:9 (NanoAOD) | 1904 | 1423 | 35 |

Baskets of 10k and 50k events read equally fast; 200k-event baskets read 30 %
slower. Splitting the same events into 1, 4 or 16 files changes nothing on
local disk (0.06 s per open); over the WAN each open costs 0.5-2 s, so parts
should hold roughly a million events.

Conclusions: analysis runs that stream from Wisconsin are bandwidth bound,
so bytes per event is what matters there; LZMA gives the smallest files but
decompresses ten times slower, which would dominate local-disk runs with
systematics. `zstd:19` keeps zstd decompression speed at 24 % below zstd:9
in size and is the default; `lzma:9` is the choice if storage is the only
concern. Parts are capped by size (2 GB, about 0.9 M MC events) rather than
by event count: size is what transfers, opens and retries see, while the
analyzer's chunking already bounds memory per task. One file per dataset was
rejected because a single failed transfer or corrupt file would cost tens of
GB and because uploads and validation are per part.
