CREATE TABLE "models" (
  "id" INTEGER PRIMARY KEY,
  "architecture" TEXT,
  "instance" TEXT,
  "object_source" TEXT
);

CREATE TABLE "failure_modes" (
	"id" INTEGER PRIMARY KEY,
  "name" TEXT
);

CREATE TABLE "distributions" (
  "id" INTEGER PRIMARY KEY,
  "name" TEXT
);

CREATE TABLE "outputs" (
  "id" INTEGER PRIMARY KEY ,
  "model" INTEGER,
  "sentence" INTEGER,
  "distribution" INTEGER,
  "detections" blob,
  "probabilities" blob,
  "failure_mode" int,
  "split" TEXT,
  FOREIGN KEY (model) REFERENCES models(id),
	FOREIGN KEY (sentence) REFERENCES sentences(id),
  FOREIGN KEY (distribution) REFERENCES distributions(id),
  FOREIGN KEY (failure_mode) REFERENCES failure_modes(id)
);

CREATE TABLE "targets" (
  "id" INTEGER PRIMARY KEY ,
  "tlx" real,
  "tly" real,
  "brx" real,
  "bry" real,
  "image_loc" TEXT
);

CREATE TABLE "sentences" (
  "id" INTEGER PRIMARY KEY ,
  "phrase" TEXT,
  "phrase_formatted" TEXT,
  "target" id,
	FOREIGN KEY (target) references targets(id)
);

CREATE INDEX IF NOT EXISTS index_outputs_model ON outputs (model);
CREATE INDEX IF NOT EXISTS index_outputs_sentence ON outputs (sentence);
CREATE INDEX IF NOT EXISTS index_outputs_distribution ON outputs (distribution);
CREATE INDEX IF NOT EXISTS index_outputs_failure_mode ON outputs (failure_mode);
CREATE INDEX IF NOT EXISTS index_sentences_target ON sentences (target);
