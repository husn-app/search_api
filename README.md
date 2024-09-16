### Myntra Image Loading
1. Loads the image on the products page using `/f_webp,dpr_2.0,q_60,w_210,c_limit,fl_progressive/assets`
    - f_webp,q_60 means there's lossy webp compression. q_100 takes 81 KB, q_95 50KB, q_80 18KB, q_60 takes 12KB. Default is q_80
    - without f_webp,q_60 takes 30s. without f_webp, for the same q_* the size is higher, for example q_60 is 20KB, q_80 is 30KB. so it's probably using some other compression method as default. 
2. Loads the image on single product page using `/h_720,q_90,w_540`.
    - doesn't use f_webp somehow even thouogh it's smaller and difference is not noticeable. 
