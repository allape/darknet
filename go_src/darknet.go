package darknet

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path"
	"strings"
	"sync"
	"time"
)

type Data struct {
	Message string
	Code    int
}

type DataChannel chan Data

type FilePath string
type FolderPath string

type IsShuttingDownGetter func() bool

type Box struct {
	Label  string `json:"label"`  // aka class name
	Left   uint   `json:"left"`   // aka x1
	Top    uint   `json:"top"`    // aka y1
	Right  uint   `json:"right"`  // aka x2
	Bottom uint   `json:"bottom"` // aka y2
}

// TruthTraining: data for CNN training
type TruthTraining struct {
	// Image: to use this, there must be a text file named fmt.Sprintf("%s.txt", TruthTraining.Image) which contains boxes
	// see: /Users/xigua/Documents/Playground/darknet/src/data.c:447, fill_truth_detection
	// boxes syntax see: /Users/xigua/Documents/Playground/darknet/scripts/voc_label.py
	// see Box2Truth
	Image FilePath `json:"image"`
	Boxes []Box    `json:"boxes"`
}

// Guess: data for predicating
type Guess struct {
	Image FilePath `json:"image"` // the path of an image, on the local machine
}

type Predication struct {
	Probability float64 `json:"probability"`
	Box
}

var (
	DarknetRoot FolderPath
	DataCfg     FilePath
	NetworkCfg  FilePath
	WeightFile  FilePath
)

var (
	TrainerLocker sync.Locker = &sync.Mutex{}
	TrainerCmd    *exec.Cmd   = nil
	TrainerIn     DataChannel = make(DataChannel)
	TrainerOut    DataChannel = make(DataChannel)
	Training      bool        = false
)

var (
	PredicatorLocker sync.Locker = &sync.Mutex{}
	PredicatorCmd    *exec.Cmd   = nil
	PredicatorIn     DataChannel = make(DataChannel)
	PredicatorOut    DataChannel = make(DataChannel)
	Predicating      bool        = false
)

func SetupTrainer(isShuttingDownGetter IsShuttingDownGetter) {
	TrainerLocker.Lock()
	defer TrainerLocker.Unlock()
	if TrainerCmd != nil {
		log.Println("Re-setup trainer")
		TrainerIn <- Data{
			Code: -1,
		}
		_ = TrainerCmd.Process.Kill()
	}
	TrainerCmd = exec.Command(
		path.Join(string(DarknetRoot), "darknet"),
		"detector",
		"train",
		string(DataCfg),
		string(NetworkCfg),
		string(WeightFile),
	)
	SetupAsyncCommand(
		"trainer",
		TrainerCmd,
		TrainerIn,
		TrainerOut,
		func() bool {
			return Training
		},
	)
	go func() {
		err := TrainerCmd.Wait()
		log.Println("trainer exited with:", err)
		if !isShuttingDownGetter() {
			time.Sleep(3 * time.Second)
			SetupTrainer(isShuttingDownGetter)
		}
	}()
}

func SetupPredicator(isShuttingDownGetter IsShuttingDownGetter) {
	PredicatorLocker.Lock()
	defer PredicatorLocker.Unlock()
	if PredicatorCmd != nil {
		PredicatorIn <- Data{
			Code: -1,
		}
		log.Println("Re-setup predicator")
		_ = PredicatorCmd.Process.Kill()
	}
	PredicatorCmd = exec.Command(
		path.Join(string(DarknetRoot), "darknet"),
		"detect",
		string(DataCfg),
		string(NetworkCfg),
		string(WeightFile),
	)
	SetupAsyncCommand(
		"predicator",
		PredicatorCmd,
		PredicatorIn,
		PredicatorOut,
		func() bool {
			return Predicating
		},
	)
	go func() {
		err := PredicatorCmd.Wait()
		log.Println("predicator exited with:", err, PredicatorCmd.ProcessState)
		if !isShuttingDownGetter() {
			time.Sleep(3 * time.Second)
			SetupPredicator(isShuttingDownGetter)
		}
	}()
}

func SetupAsyncCommand(
	name string,
	cmd *exec.Cmd,
	incomingChan DataChannel,
	outgoingChan DataChannel,
	isChannelOnline func() bool,
) {
	stdin, err := cmd.StdinPipe()
	if err != nil {
		SendToChannel(fmt.Sprintf("Unable to open stdin of %s because (of) %v", name, err), outgoingChan, isChannelOnline)
		return
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		SendToChannel(fmt.Sprintf("Unable to open stdout of %s because (of) %v", name, err), outgoingChan, isChannelOnline)
		return
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		SendToChannel(fmt.Sprintf("Unable to open stderr of %s because (of) %v", name, err), outgoingChan, isChannelOnline)
		return
	}

	err = cmd.Start()
	if err != nil {
		fmt.Printf("Unable to start %s because (of) %v", name, err)
		return
	}

	go func() {
		for {
			d := <-incomingChan
			if d.Code == -1 {
				break
			}
			data := d.Message
			log.Println(name+":stdin", "<", data)

			bs := []byte(data)
			n, err := io.Copy(stdin, bytes.NewBufferString(data))
			if err != nil {
				SendToChannel(fmt.Sprintf("Unable to write into stdin of %s because (of) %v", name, err), outgoingChan, isChannelOnline)
				continue
			}
			if int(n) != len(bs) {
				SendToChannel(fmt.Sprintf("Expected length of written bytes is %d but got %d in %s", len(bs), n, name), outgoingChan, isChannelOnline)
				continue
			}
		}
	}()
	go AsyncReader(stdout, name+":stdout", outgoingChan, isChannelOnline)
	go AsyncReader(stderr, name+":stderr", outgoingChan, isChannelOnline)
}

func AsyncReader(reader io.Reader, name string, outChan DataChannel, isChannelOnline func() bool) {
	lastUnfinishedLine := ""
	buff := make([]byte, 4096)
	done := false
	for {
		n, err := reader.Read(buff)
		out := string(buff[:n])
		log.Println(name, ">", out)
		if err != nil {
			SendToChannel(fmt.Sprintf("Unable to read from %s because (of) %v", name, err), outChan, isChannelOnline)
			if err == io.EOF {
				log.Println("EOF on", name)
				done = true
			} else {
				continue
			}
		}
		lines := strings.Split(out, "\n")
		lineCount := len(lines)
		if lineCount == 1 {
			lastUnfinishedLine += lines[0]
			continue
		}
		secondIndexToLastLine := lineCount - 2
		for index, line := range lines {
			if index == 0 && lastUnfinishedLine != "" {
				line = lastUnfinishedLine + line
				lastUnfinishedLine = ""
			}

			SendToChannel(line, outChan, isChannelOnline)

			if secondIndexToLastLine == index {
				lastLine := lines[index+1]
				if lastLine != "" {
					lastUnfinishedLine = lastLine
				}
				break
			}
		}
		if done {
			if lastUnfinishedLine != "" {
				SendToChannel(lastUnfinishedLine+"\n", outChan, isChannelOnline)
			}
			break
		}
	}
}

func CloseChannel(channel chan string) {
	if channel != nil {
		close(channel)
	}
}

func SendToChannel(data string, channel DataChannel, isChannelOnline func() bool) {
	log.Println(data)
	if !isChannelOnline() {
		log.Println("Channel is not online for now")
		return
	}
	channel <- Data{
		Message: data,
	}
}

// Box2Truth
// see /Users/xigua/Documents/Playground/darknet/scripts/voc_label.py
func Box2Truth(imageWidth int, imageHeight int, box Box) (float64, float64, float64, float64) {
	dw, dh := 1./float64(imageWidth), 1./float64(imageHeight)
	var x float64 = float64(box.Left+box.Right)/2.0 - 1
	var y float64 = float64(box.Top+box.Bottom)/2.0 - 1
	w, h := float64(box.Right-box.Left), float64(box.Bottom-box.Top)
	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh
	return x, y, w, h
}

func ClassNamesFromDataConfig(dataCfg FilePath) ([]string, error) {
	file, err := os.ReadFile(string(dataCfg))
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(file), "\n")
	for _, line := range lines {
		config := strings.Split(line, "=")
		if len(config) < 2 {
			continue
		}
		if strings.TrimSpace(config[0]) == "names" {
			nameFileName := strings.TrimSpace(config[1])
			names, err := os.ReadFile(path.Join(string(DarknetRoot), nameFileName))
			if err != nil {
				return nil, err
			}
			return strings.Split(string(names), "\n"), nil
		}
	}
	return nil, nil
}
