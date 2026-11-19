// Command tags writes the part-of-speech tags of every document in a corpus
// directory, one JSON line per document, for script/features.py to read.
//
// The tagger is prose's, the one Vale's `sequence` rules run on, so a
// feature built from these tags is one a rule can test.
//
// Usage: go run . <dir> [<dir>...]   (writes <dir>/tags.jsonl)
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"

	"github.com/jdkato/prose/v3"
)

type line struct {
	Name string   `json:"name"`
	Tags []string `json:"tags"`
}

func main() {
	for _, dir := range os.Args[1:] {
		files, _ := filepath.Glob(filepath.Join(dir, "*.md"))
		sort.Strings(files)

		out := make([]line, len(files))
		jobs := make(chan int)
		var wg sync.WaitGroup
		for range runtime.NumCPU() {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range jobs {
					text, err := os.ReadFile(files[i])
					if err != nil {
						continue
					}
					doc, err := prose.NewDocument(string(text),
						prose.WithExtraction(false), prose.WithSegmentation(false))
					if err != nil {
						continue
					}
					tags := make([]string, 0, 256)
					for _, tok := range doc.Tokens() {
						tags = append(tags, tok.Tag)
					}
					out[i] = line{Name: filepath.Base(files[i]), Tags: tags}
				}
			}()
		}
		for i := range files {
			jobs <- i
		}
		close(jobs)
		wg.Wait()

		f, err := os.Create(filepath.Join(dir, "tags.jsonl"))
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		w := bufio.NewWriter(f)
		enc := json.NewEncoder(w)
		for _, l := range out {
			if l.Name != "" {
				_ = enc.Encode(l)
			}
		}
		w.Flush()
		f.Close()
		fmt.Printf("%s: %d documents\n", dir, len(files))
	}
}
