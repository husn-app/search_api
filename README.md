### Myntra Image Loading
1. Loads the image on the products page using `/f_webp,dpr_2.0,q_60,w_210,c_limit,fl_progressive/assets`
    - f_webp,q_60 means there's lossy webp compression. q_100 takes 81 KB, q_95 50KB, q_80 18KB, q_60 takes 12KB. Default is q_80
    - without f_webp,q_60 takes 30s. without f_webp, for the same q_* the size is higher, for example q_60 is 20KB, q_80 is 30KB. so it's probably using some other compression method as default. 
2. Loads the image on single product page using `/h_720,q_90,w_540`.
    - doesn't use f_webp somehow even thouogh it's smaller and difference is not noticeable. 


### Retrieval Models
1. Retrieval results for some of the famous models including SigLiP are in [open_clip/openclip_retrieval_results](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_retrieval_results.csv), sorted results are in this [doc](https://docs.google.com/spreadsheets/d/1ilPJexX2m03QtX74iaeGCdBQ3sVm5jYqQ0Kv2BgrDe0/edit?gid=1066211703#gid=1066211703)
2. There are two heads one for image and one for text. We can load only the text module for now.
3. There's a much smaller mobile-clip variant from apple : https://github.com/apple/ml-mobileclip
4. This [tweet](https://x.com/giffmana/status/1717999891937394990) from Lucas says there is a 87M clip as well, which performs exceptionally, but I haven't found this variant anywhere.
5. Note that we also need to optimize for number of dimensions in the output, since that will take up latency during retrieval. The current Vit-B-32 has 512 dimensions, but some siglip variants have higher like 768. 
