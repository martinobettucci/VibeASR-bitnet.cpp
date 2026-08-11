/**
 * Decoder-level hotword biasing (shallow contextual biasing).
 *
 * Prompt injection was measured and rejected for this purpose: putting terms in the
 * prompt reconditions the whole distribution and took French from 36% to 90% WER
 * (oracle terms!). This does the standard thing instead: a token-level trie over the
 * hotword list, and a bounded bonus added to the logits of exactly the tokens that
 * would continue a match, immediately before sampling. The model's distribution is
 * otherwise untouched; the thumb only rests on the scale where a listed term is a
 * live continuation.
 *
 * Greedy decoding keeps the machinery small: one hypothesis, one cursor. The cursor
 * follows matched arcs; on a miss it falls back to the root (a new match may start
 * at any token). Root children are boosted alongside cursor children so multi-word
 * matches can begin mid-utterance.
 *
 * BPE reality: "Martelly" tokenizes differently at a word boundary than mid-word, so
 * every entry is inserted in its raw, space-prefixed, lowercase and space-lowercase
 * variants.
 */

#ifndef HOTWORD_BOOST_H
#define HOTWORD_BOOST_H

#include "llama.h"
#include "prompt_builder.h"

#include <cctype>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace hotword {

struct Trie {
    // node 0 is the root; children[n] maps token -> child node
    std::vector<std::map<llama_token, int> > children;
    float lambda = 0.0f;

    bool active() const { return lambda != 0.0f && children.size() > 1; }

    void insert(const std::vector<llama_token> & seq) {
        int cur = 0;
        for (size_t i = 0; i < seq.size(); i++) {
            std::map<llama_token, int>::iterator it = children[cur].find(seq[i]);
            if (it == children[cur].end()) {
                children.push_back(std::map<llama_token, int>());
                children[cur][seq[i]] = (int) children.size() - 1;
                cur = (int) children.size() - 1;
            } else {
                cur = it->second;
            }
        }
    }
};

inline Trie build(const llama_model * model, const std::string & csv, float lambda) {
    Trie t;
    t.children.push_back(std::map<llama_token, int>());
    t.lambda = lambda;

    size_t pos = 0;
    while (pos <= csv.size()) {
        size_t comma = csv.find(',', pos);
        if (comma == std::string::npos) comma = csv.size();
        std::string w = csv.substr(pos, comma - pos);
        pos = comma + 1;
        // trim
        while (!w.empty() && isspace((unsigned char) w.front())) w.erase(w.begin());
        while (!w.empty() && isspace((unsigned char) w.back())) w.pop_back();
        if (w.empty()) continue;

        std::string lower = w;
        for (size_t i = 0; i < lower.size(); i++) lower[i] = (char) tolower((unsigned char) lower[i]);

        std::set<std::string> variants;
        variants.insert(w);
        variants.insert(" " + w);
        variants.insert(lower);
        variants.insert(" " + lower);

        for (std::set<std::string>::iterator v = variants.begin(); v != variants.end(); ++v) {
            std::vector<llama_token> toks = prompt_builder::tokenize(model, *v, false, false);
            if (!toks.empty()) t.insert(toks);
        }
    }
    return t;
}

// Decode-time state: the deepest node reached by the tokens emitted so far.
struct State {
    int cursor = 0;

    // Add lambda to every token that extends a live match. Cursor children first,
    // then root children (a fresh match can start anywhere); a token in both sets is
    // boosted once.
    void boost(const Trie & t, float * logits) const {
        if (!t.active()) return;
        const std::map<llama_token, int> & cur = t.children[cursor];
        for (std::map<llama_token, int>::const_iterator it = cur.begin(); it != cur.end(); ++it) {
            logits[it->first] += t.lambda;
        }
        if (cursor != 0) {
            const std::map<llama_token, int> & root = t.children[0];
            for (std::map<llama_token, int>::const_iterator it = root.begin(); it != root.end(); ++it) {
                if (cur.find(it->first) == cur.end()) logits[it->first] += t.lambda;
            }
        }
    }

    void advance(const Trie & t, llama_token tok) {
        if (!t.active()) return;
        std::map<llama_token, int>::const_iterator it = t.children[cursor].find(tok);
        if (it != t.children[cursor].end()) { cursor = it->second; return; }
        it = t.children[0].find(tok);
        cursor = it != t.children[0].end() ? it->second : 0;
    }
};

}  // namespace hotword

#endif  // HOTWORD_BOOST_H
