
from dataclasses import dataclass
import re
from typing import List

from .wikis import WikiLookUp


@dataclass
class Evidence:
    annotation_id: int
    evidence_id: int
    wikipedia_url: str = None
    sentence_id: str = None
    sentence: str = None

    @classmethod
    def from_list(cls, ev: list, wklp: WikiLookUp):
        if ev[2] is not None and ev[3] is not None:
            text = wklp.find_text_by_url(ev[2])
            sentence = Evidence.get_sentence_from_id(text, ev[3])
        else:
            sentence = ""
        return cls(
            annotation_id=ev[0],
            evidence_id=ev[1],
            wikipedia_url=ev[2],
            sentence_id=ev[3],
            sentence=sentence
        )

    @staticmethod
    def get_sentence_from_id(text: str, sentence_id: int) -> str:
        # sentences = re.findall("\d\\t(.+?[.])\\t", text)
        # sentences = sent_tokenize(text)
        if text is None:
            return ""
        sentences = text.split("\n")
        # try:
        return sentences[sentence_id].split("\t")[1]
        # except IndexError:
        #     print(text)
        #     print(text.split("\n"))
        #     print(sentence_id, len(sentences))
        #     exit()


@dataclass
class EvidenceList:
    evidences: List[Evidence]

    @classmethod
    def from_list(cls, evs: List[list], wklp: WikiLookUp):
        evidences = []
        for ev in evs:
            for s in ev:
                evidences.append(Evidence.from_list(s, wklp))
        return cls(evidences=evidences)

    def to_text(self) -> str:
        sentences = []
        for ev in self.evidences:
            if ev.sentence and ev.sentence not in sentences:
                sentences.append(ev.sentence)
        return " ".join(sentences) if sentences else ""
