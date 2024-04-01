The code of WWW2023 paper "Contrastive Collaborative Filtering for Cold-Start Item Recommendation"

# Data Note:
* image_features
* negative_user: For each interaction, we need one negative user which is not interacted with the item.
* negative_item: For each item, we need to get multiple negative items which are not interacted with the same user as the positive item.
* multi-hot features vector:
    * MovieLens: 18 genres, each vector is a 18-dim vector, if the movie belongs to the genre, the corresponding position is equal to index of that position, otherwise, it is 18. [0, 18, 18, 18, 4, 18, 18, 18, 18, 18, 10, 18, 12, 18, 18, 18, 18, 18]
    * Amazon: {item_id, [categrory_id1, category_id2, ...]}
