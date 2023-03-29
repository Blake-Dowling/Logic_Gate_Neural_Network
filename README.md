Increments each parameter by equal amount, which is the amount specified as a parameter (INC_AMOUNT=) when creating the Neural object.

Setting the parameter increment amount to a small nonnegative value is necessary for accurate results. Setting the target phi value to a small number achieves this as well, but degrades performance.

Obstacles: Understanding how the neural network is supposed to learn, particularly regarding deep learning. The activation function transforms output between layers non-linearly.

Noticed a significant improvement in accuracy when initializing weights to small nonnegative numbers. This also resolved the issue of becoming stuck in a loop when switching between gate types.

Utilization may be imporoved by multithreading the feed function, which is run once for each parameter adjustment in order to calculate phi. If this is done, the neural object's parameters must not be changed until after the loss function is performed, and the feed function must be able to take a copy of an argument vector as a parameter instead of using the nerual object's set parameters.

Trials are showing tradeoff between precision and time complexity.


https://user-images.githubusercontent.com/121590227/228456592-50339c19-1a5d-49ef-b02e-1139206051a8.mov


https://user-images.githubusercontent.com/121590227/228456001-f17d56e6-1b43-4e0a-b656-46a7cc3f6c0f.mov

<img width="641" alt="Screenshot 2023-03-29 at 12 14 33 AM" src="https://user-images.githubusercontent.com/121590227/228455548-9c13785f-8509-4c1c-aae7-1f8ff2ac9f31.png">
