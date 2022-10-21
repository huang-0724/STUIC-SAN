# STUIC-SAN: Spatio-temporal Unequal Interval Correlation-aware Self-Attention Network for Next POI Recommendation

As the core of Location-based Social Networks (LBSNs), the main task of next Point-of-Interest (POI) recommendation is
 to predict next possible POI through the context information from users historical check-in trajectories. It is well known 
that spatial-temporal contextual information plays an important role in analyzing users check-in behaviors. Moreover, the
 information between POIs provides a non-trivial correlation for modeling users visiting preferences. Unfortunately, the 
impact of such correlation information and the spatio-temporal unequal interval information between POIs on user selection 
of next POI, is rarely considered.  Therefore, we propose a Spatio-Temporal Unequal Interval Correlation-aware Self-Attention
 Network (STUIC-SAN) model for next POI recommendation. Specifically, we first use the linear regression method to obtain the 
spatio-temporal unequal interval correlation between any two POIs from users check-in sequences. Sequentially, we design
 a spatio-temporal unequal interval correlation-aware self-attention mechanism, which is able to comprehensively capture users personalized 
spatio-temporal unequal interval correlation preferences by incorporating multiple factors including POIs information, 
spatio-temporal unequal interval correlation information between POIs, and absolute positional information of corresponding
 POIs . On this basis, we perform next POI recommendation. Finally, we conducted 
comprehensive performance evaluation using large-scale real-world datasets from two popular
 location-based social networks, namely, Foursquare and Gowalla. Experimental results on two datasets indicated that 
the proposed STUIC-SAN outperformed the state-of-the-art next POI recommendation approaches regarding two 
commonly-used evaluation metrics.

## Environment
Python 3.6

TensorFlow 1.14.0

Numpy 1.16.0

## Datasets

This repo includes ml-1m dataset as an example.

For Foursquare dataset, you could download Foursquare data from *[here.](https://developer.foursquare.com/places-api)*.

## Model Training

To train our model on `foursquare data` (with default hyper-parameters): 

```
python main.py --dataset=foursquare data --train_dir=default 
```



