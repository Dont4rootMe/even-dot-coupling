import numpy as np
import matplotlib.pyplot as plt


def solve_complete_matching_robust(
    A_coords: np.ndarray,
    B_coords: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Compute a guaranteed non-crossing perfect matching between two planar point sets.

    The algorithm is a recursive ham-sandwich style angular sweep around a pivot.
    For two balanced point sets in general position it produces a non-intersecting
    matching. Worst-case complexity is O(N^2 log N) due to the per-level sorting.
    """
    # Structured array keeps coordinates, original index and set type together
    # type: +1 (A), -1 (B)
    point_dtype = [
        ("coords", float, (2,)),
        ("id", int),
        ("type", int),
    ]

    n = len(A_coords)
    points = np.zeros(2 * n, dtype=point_dtype)

    # Fill A points
    points["coords"][:n] = A_coords
    points["id"][:n] = np.arange(n)
    points["type"][:n] = 1

    # Fill B points
    points["coords"][n:] = B_coords
    points["id"][n:] = np.arange(n)
    points["type"][n:] = -1

    matches: list[tuple[int, int]] = []

    def recursive_solver(current_set: np.ndarray) -> None:
        if len(current_set) == 0:
            return

        # 1. Pivot: global minimum by y, then by x
        min_idx = np.lexsort(
            (current_set["coords"][:, 0], current_set["coords"][:, 1])
        )[0]
        pivot = current_set[min_idx]

        # Remove pivot from the working set
        others = np.delete(current_set, min_idx)

        # 2. Polar angles relative to pivot
        vecs = others["coords"] - pivot["coords"]
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])

        # 3. Sort by angle
        sort_order = np.argsort(angles)
        sorted_others = others[sort_order]

        # 4. Angular scan to find a partner:
        # we need a point of opposite type such that the prefix before it
        # contains equal numbers of A and B (balance == 0)
        balance = 0
        split_idx = -1
        pivot_type = pivot["type"]

        for i, target in enumerate(sorted_others):
            if target["type"] == -pivot_type and balance == 0:
                split_idx = i
                break

            if target["type"] == pivot_type:
                balance += 1
            else:
                balance -= 1

        if split_idx == -1:
            # For balanced sets in general position this should not happen
            raise ValueError("Geometric invariant violated: split not found.")

        partner = sorted_others[split_idx]

        # Store pair as (index_A, index_B)
        if pivot_type == 1:
            matches.append((pivot["id"], partner["id"]))
        else:
            matches.append((partner["id"], pivot["id"]))

        # 5. Recurse on the two sides of the cut
        left_set = sorted_others[:split_idx]
        right_set = sorted_others[split_idx + 1 :]

        recursive_solver(left_set)
        recursive_solver(right_set)

    recursive_solver(points)
    return matches


def plot_matching(
    A: np.ndarray,
    B: np.ndarray,
    matches: list[tuple[int, int]],
) -> None:
    """Visualize the resulting non-crossing matching."""
    plt.figure(figsize=(8, 8))

    # Draw segments
    for idx_a, idx_b in matches:
        pa = A[idx_a]
        pb = B[idx_b]
        plt.plot([pa[0], pb[0]], [pa[1], pb[1]], "k-", alpha=0.8, zorder=1)

    # Draw points
    plt.scatter(
        A[:, 0],
        A[:, 1],
        c="red",
        s=80,
        edgecolors="white",
        label="A",
        zorder=2,
    )
    plt.scatter(
        B[:, 0],
        B[:, 1],
        c="blue",
        s=80,
        edgecolors="white",
        label="B",
        zorder=2,
    )

    plt.title(f"Non-intersecting Matching (N={len(A)})")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    n_points = 15
    A = np.random.rand(n_points, 2)
    B = np.random.rand(n_points, 2)

    try:
        result_pairs = solve_complete_matching_robust(A, B)
        print(f"Resulting pairs (idx_A, idx_B): {result_pairs}")
        plot_matching(A, B, result_pairs)
    except Exception as e:
        print(f"Matching failed: {e}")
