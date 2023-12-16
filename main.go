package main

import (
	darknet "darknet/go_src"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path"
	"slices"
	"strconv"
	"strings"
	"syscall"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

var (
	TemporaryImageFolder string
	ShuttingDown         bool // 是否在关闭状态
)

var AcceptableExts = []string{
	".jpg",
	".jpeg",
	".png",
}

func isShuttingDown() bool {
	return ShuttingDown
}

func multipart2image(ctx *gin.Context) (darknet.FilePath, *os.File, int, error) {
	form, err := ctx.MultipartForm()
	if err != nil {
		return "", nil, http.StatusBadRequest, err
	}
	images, ok := form.File["image"]
	if !ok || len(images) == 0 {
		return "", nil, http.StatusBadRequest, errors.New("No image found")
	}
	if len(images) > 1 {
		return "", nil, http.StatusBadRequest, errors.New("Only one image is acceptable")
	}
	image := images[0]
	if !slices.Contains[[]string](AcceptableExts, path.Ext(image.Filename)) {
		return "", nil, http.StatusBadRequest, fmt.Errorf("File %s is not acceptable, expected %v", image.Filename, AcceptableExts)
	}
	openImage, err := image.Open()
	if err != nil {
		return "", nil, http.StatusInternalServerError, fmt.Errorf("Unable to open this multipart file: %v", err)
	}
	defer func() {
		_ = openImage.Close()
	}()
	tmpFilePath := path.Join(TemporaryImageFolder, fmt.Sprintf("yolor_%s", image.Filename))
	tmpFile, err := os.Create(tmpFilePath)
	if err != nil {
		return darknet.FilePath(tmpFilePath), tmpFile, http.StatusInternalServerError, fmt.Errorf("Unable to create temp file: %v", err)
	}
	n, err := io.Copy(tmpFile, openImage)
	if err != nil {
		return darknet.FilePath(tmpFilePath), tmpFile, http.StatusInternalServerError, fmt.Errorf("Unable to copy file from multipart form to temp file: %v", err)
	}
	if n != image.Size {
		return darknet.FilePath(tmpFilePath), tmpFile, http.StatusInternalServerError, fmt.Errorf("Temp file is not identical with the image from multipart form: size is not as the same as copied")
	}
	return darknet.FilePath(tmpFilePath), tmpFile, http.StatusOK, nil
}

func main() {
	cwd, _ := os.Getwd()

	darknet.DarknetRoot = env[darknet.FolderPath]("YOLOR_DARKNET_ROOT", darknet.FolderPath(cwd))
	darknet.DataCfg = env[darknet.FilePath]("YOLOR_DATA_CFG", "cfg/coco.data")
	darknet.NetworkCfg = env[darknet.FilePath]("YOLOR_NETWORK_CFG", "cfg/yolov3.cfg")
	darknet.WeightFile = env[darknet.FilePath]("YOLOR_WEIGHT_FILE", "weights/yolov3.weights")

	if !isFolderOk(darknet.DarknetRoot) {
		panic(fmt.Sprintf("DarknetRoot %s is not valid", darknet.DarknetRoot))
	}
	if !isFileOk(darknet.DataCfg) {
		panic(fmt.Sprintf("DataCfg %s is not valid", darknet.DataCfg))
	}
	if !isFileOk(darknet.NetworkCfg) {
		panic(fmt.Sprintf("NetworkCfg %s is not valid", darknet.NetworkCfg))
	}
	if !isFileOk(darknet.WeightFile) {
		panic(fmt.Sprintf("WeightFile %s is not valid", darknet.WeightFile))
	}

	// extract names from data config
	names, err := darknet.ClassNamesFromDataConfig(darknet.DataCfg)
	if err != nil {
		panic(fmt.Sprintf("Unable to extract names from data cfg %s because (of) %v", darknet.DataCfg, err))
	}

	TemporaryImageFolder = path.Join(cwd, "images")
	statOfTIF, err := os.Stat(TemporaryImageFolder)
	if err != nil {
		if os.IsNotExist(err) {
			os.MkdirAll(TemporaryImageFolder, 0o777)
		} else {
			panic(fmt.Sprintf("Unable to create temporary image folder: %v", err))
		}
	} else if !statOfTIF.IsDir() {
		panic(fmt.Sprintf("Temporary image folder should be a dir"))
	}

	// darknet.go SetupTrainer(isShuttingDown)
	go darknet.SetupPredicator(isShuttingDown)

	router := gin.Default()

	router.GET("/names", func(ctx *gin.Context) {
		ctx.JSON(http.StatusOK, names)
	})
	router.POST("/predicate", HandlePredicate)

	router.Use(cors.Default())
	go func() {
		_ = router.Run()
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM, syscall.SIGABRT)
	sig := <-sigChan
	ShuttingDown = true
	fmt.Println("\nReceived signal:", sig)
	if darknet.TrainerCmd != nil {
		_ = darknet.TrainerCmd.Process.Kill()
	}
	if darknet.PredicatorCmd != nil {
		_ = darknet.PredicatorCmd.Process.Kill()
	}
	os.Exit(0)
}

func HandlePredicate(ctx *gin.Context) {
	darknet.PredicatorLocker.Lock()
	darknet.Predicating = true
	defer func() {
		darknet.PredicatorLocker.Unlock()
		darknet.Predicating = false
	}()

	imagePath, _, statusCode, err := multipart2image(ctx)
	defer func() {
		_ = os.Remove(string(imagePath))
	}()
	if err != nil {
		ctx.JSON(statusCode, err)
		return
	}

	darknet.PredicatorIn <- darknet.Data{
		Message: string(imagePath) + "\n",
	}
	var lines []string
	for {
		data := <-darknet.PredicatorOut
		line := data.Message
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, "yolor: Done predicating") {
			break
		}
		lines = append(lines, line)
	}
	// yolor:boxes:person,93.28,249,205,261,241,person,90.11,272,205,287,241,person,82.86,214,197,222,219
	//              label, prob, x1, y1, x2, y2
	Token := "yolor:boxes:"
	ColumnGroupCount := 6
	predicationsString := lines[len(lines)-1]
	if strings.HasPrefix(predicationsString, Token) && strings.Contains(predicationsString, ",") {
		predicationsString = strings.TrimSuffix(predicationsString[len(Token):], ",")
		columns := strings.Split(predicationsString, ",")
		if len(columns)%ColumnGroupCount != 0 {
			ctx.JSON(http.StatusInternalServerError, "Results are not match")
			return
		}
		predicationCount := len(columns) / ColumnGroupCount
		predications := make([]darknet.Predication, predicationCount)
		for i := 0; i < predicationCount; i++ {
			columnIndex := i * ColumnGroupCount
			prob, _ := strconv.ParseFloat(columns[columnIndex+1], 64)
			left, _ := strconv.ParseUint(columns[columnIndex+2], 10, 32)
			top, _ := strconv.ParseUint(columns[columnIndex+3], 10, 32)
			right, _ := strconv.ParseUint(columns[columnIndex+4], 10, 32)
			bottom, _ := strconv.ParseUint(columns[columnIndex+5], 10, 32)
			predication := darknet.Predication{
				Probability: prob,
				Box: darknet.Box{
					Label:  columns[columnIndex+0],
					Left:   uint(left),
					Top:    uint(top),
					Right:  uint(right),
					Bottom: uint(bottom),
				},
			}
			predications[i] = predication
		}
		ctx.JSON(http.StatusOK, predications)
	} else {
		ctx.JSON(http.StatusOK, []string{})
	}
}

func HandleTrain(ctx *gin.Context) {
	darknet.TrainerLocker.Lock()
	darknet.Training = true
	defer func() {
		darknet.Training = false
		darknet.TrainerLocker.Unlock()
	}()

	imagePath, _, statusCode, err := multipart2image(ctx)
	defer func() {
		_ = os.Remove(string(imagePath))
	}()
	if err != nil {
		ctx.JSON(statusCode, err)
		return
	}
}

func isFolderOk(filename darknet.FolderPath) bool {
	file, err := os.Stat(string(filename))
	return err == nil && file.IsDir()
}

func isFileOk(filename darknet.FilePath) bool {
	file, err := os.Stat(string(filename))
	return err == nil && !file.IsDir()
}

func env[T ~string](key string, defaultValue T) T {
	val := os.Getenv(key)
	if val == "" {
		return defaultValue
	}
	return T(val)
}
