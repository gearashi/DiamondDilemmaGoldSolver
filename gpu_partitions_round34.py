"""Exact GPU queue splitting by the smallest unresolved cell in the largest cube."""
import heapq
import math


def split_small_domains(cubes, count):
    cubes = list(cubes)
    if len(cubes) >= count:
        return cubes
    heap = [(-math.prod(mask.bit_count() for mask in cube), i, tuple(cube))
            for i, cube in enumerate(cubes)]
    heapq.heapify(heap)
    ticket = len(heap)
    while len(heap) < count and heap and heap[0][0] < -1:
        negative_size, _, row = heapq.heappop(heap)
        width, cell = min((mask.bit_count(), i) for i, mask in enumerate(row)
                          if mask.bit_count() > 1)
        right, left = row[cell], 0
        for _ in range(width // 2):
            bit = right & -right
            left |= bit
            right ^= bit
        factor = -negative_size // width
        for part in (left, right):
            child = row[:cell] + (part,) + row[cell + 1:]
            heapq.heappush(heap, (-factor * part.bit_count(), ticket, child))
            ticket += 1
    return [entry[2] for entry in heap]
