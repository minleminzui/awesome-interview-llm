from typing import List
import collections
class Solution:
  
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)

        for st in strs:
          count = [0] * 26
          for ch in st:
            count[ord(ch) - ord('a')] += 1
          mp[tuple(count)].append(st)
        return list(mp.values())

if __name__ == "__main__":
  solution = Solution()
  ans = solution.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
  print(ans)