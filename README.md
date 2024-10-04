### Retrieval Models
1. Retrieval results for some of the famous models including SigLiP are in [open_clip/openclip_retrieval_results](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_retrieval_results.csv), sorted results are in this [doc](https://docs.google.com/spreadsheets/d/1ilPJexX2m03QtX74iaeGCdBQ3sVm5jYqQ0Kv2BgrDe0/edit?gid=1066211703#gid=1066211703)
2. There are two heads one for image and one for text. We can load only the text module for now.
3. There's a much smaller mobile-clip variant from apple : https://github.com/apple/ml-mobileclip
4. This [tweet](https://x.com/giffmana/status/1717999891937394990) from Lucas says there is a 87M clip as well, which performs exceptionally, but I haven't found this variant anywhere.
5. Note that we also need to optimize for number of dimensions in the output, since that will take up latency during retrieval. The current Vit-B-32 has 512 dimensions, but some siglip variants have higher like 768. 
