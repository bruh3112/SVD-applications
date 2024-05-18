from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['figure.figsize'] = [16, 8]

# Load and preprocess the image
A = imread('C:/Users/Hoang Anh/Downloads/house.jpg')
X = np.mean(A, -1)  # convert image to grayscale

# Display the original image
plt.imshow(X, cmap='gray')
plt.axis('off')
plt.show()

# Perform Singular Value Decomposition (SVD)
U, S, VT = np.linalg.svd(X, full_matrices=False)

#Image compression
S = np.diag(S)

# Display compressed images for different values of r
for r in [5, 20, 100]:
    Xapprox = U[:, :r] @ S[:r, :r] @ VT[:r, :]
    plt.figure()
    plt.imshow(Xapprox, cmap='gray')
    plt.axis('off')
    plt.title(f'Compressed Image (r={r})')
    plt.show()

# Plot singular values and cumulative sum
plt.figure()
plt.plot(range(1, 51), np.diag(S)[:50])
plt.title('Singular Values')
plt.show()

plt.figure()
plt.plot(range(1, 51), np.cumsum(np.diag(S)[:50]) / np.sum(np.diag(S)))
plt.title('Cumulative Sum of Singular Values')
plt.show()

# Plot variance explained by top singular values
var_explained = np.round(S ** 2 / np.sum(S ** 2), decimals=6)
sns.barplot(x=list(range(1, 21)), y=var_explained[:20], color="dodgerblue")
plt.title('Variance Explained Graph')
plt.xlabel('Singular Vector', fontsize=16)
plt.ylabel('Variance Explained', fontsize=16)
plt.tight_layout()
plt.show()

# Image reconstruction with different numbers of components
comps = [947, 12, 20, 30, 50, 100]
plt.figure(figsize=(12, 6))
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i in range(len(comps)):
    low_rank = U[:, :comps[i]] @ np.diag(S[:comps[i]]) @ VT[:comps[i], :]

    axes[i // 3, i % 3].imshow(low_rank, cmap='gray')
    axes[i // 3, i % 3].axis('off')
    axes[i // 3, i % 3].set_title(f'n_components = {comps[i]}')

plt.tight_layout()
plt.show()
