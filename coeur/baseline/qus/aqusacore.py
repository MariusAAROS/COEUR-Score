from coeur.baseline.qus.corefiles.wellformed import *
from coeur.baseline.qus.corefiles.analyzer import *
from coeur.baseline.qus.corefiles.globals import *
from coeur.baseline.qus.corefiles.stories import *
import numpy as np

class AQUSA:
    def __init__(self, stories: list[str]):
        self.stories = stories
    
    def compute(self) -> float:
        defects.clear()
        allStories = Stories("Dapliaz")
        for i, s in enumerate(self.stories):
            story = Story(id = i, title = s.strip())
            story = story.chunk()
            WellFormedAnalyzer.well_formed(story)
            Analyzer.atomic(story)
            MinimalAnalyzer.minimal(story)
            Analyzer.unique(story, allStories)
            allStories.add_story(story)
        allStories = Analyzer.get_common_format(allStories)
        for story in allStories.stories:
            Analyzer.uniform(story,allStories)
        output_text = ""
        for defect in defects:
            output_text = output_text + defect.print_txt()

        curated = {}
        for line in output_text.split("\n"):
            if "Story #" in line:
                story_id = int(line.split("#")[1].split(":")[0].strip())
            if "Defect type" in line:
                defect = line.split(":")[1].strip().split(".")[0].strip()
                curated.setdefault(story_id, set()).add(defect)
        
        scores = []
        for i, s in enumerate(self.stories):
            score = (5 - len(curated.get(i, []))) / 5
            scores.append(score)

        aqusa_score = np.mean(scores)
        return aqusa_score