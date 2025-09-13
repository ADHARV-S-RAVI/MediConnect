import os
import io
import json
import time
import uuid
import hashlib
import pickle
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

# Crypto
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# Data & ML
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Flask
from flask import Flask, request, jsonify

# -------------------------
# CONFIG
# -------------------------
PATIENT_KEY_STORAGE: Dict[str, bytes] = {}
METADATA_FILE = "records_metadata.json"
OFFCHAIN_FOLDER = "offchain_storage"

if not os.path.exists(OFFCHAIN_FOLDER):
    os.makedirs(OFFCHAIN_FOLDER)

# -------------------------
# Helpers
# -------------------------
def now_ts() -> float:
    return time.time()

def sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# -------------------------
# Blockchain
# -------------------------
class Block:
    def _init_(self, index: int, timestamp: str, data: dict, previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        block_string = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "data": self.data,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        )
        return sha256_hex(block_string)


class SimpleBlockchain:
    def _init_(self):
        self.chain: List[Block] = []
        self.create_genesis()

    def create_genesis(self):
        genesis = Block(
            0, datetime.now(timezone.utc).isoformat(), {"genesis": True}, "0"
        )
        self.chain.append(genesis)
        print(f"[Blockchain] Added block {genesis.index} - hash {genesis.hash[:12]}...")

    def add_block(self, data: dict) -> Block:
        prev = self.chain[-1]
        new = Block(
            prev.index + 1, datetime.now(timezone.utc).isoformat(), data, prev.hash
        )
        self.chain.append(new)
        print(f"[Blockchain] Added block {new.index} - hash {new.hash[:12]}...")
        return new

    def is_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr.previous_hash != prev.hash:
                return False
            if curr.compute_hash() != curr.hash:
                return False
        return True

# -------------------------
# Local Off-chain Storage
# -------------------------
class LocalOffChainStore:
    def _init_(self, folder=OFFCHAIN_FOLDER):
        self.folder = folder

    def store_bytes(self, content_bytes: bytes, filename_hint: str = None) -> str:
        filename = filename_hint or f"{uuid.uuid4().hex}.bin"
        path = os.path.join(self.folder, filename)
        with open(path, "wb") as f:
            f.write(content_bytes)
        print(f"[OffChain] Stored file: {path}")
        return filename

    def retrieve_bytes(self, file_id: str) -> Optional[bytes]:
        path = os.path.join(self.folder, file_id)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

# -------------------------
# Encrypted Data Store
# -------------------------
class EncryptedDataStore:
    def _init_(self, offchain_store: LocalOffChainStore):
        self.offchain = offchain_store
        if not os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "w") as f:
                json.dump({}, f)

    def generate_patient_key(self, patient_id: str) -> bytes:
        key = Fernet.generate_key()
        PATIENT_KEY_STORAGE[patient_id] = key
        return key

    def get_patient_key(self, patient_id: str) -> Optional[bytes]:
        return PATIENT_KEY_STORAGE.get(patient_id)

    def _load_metadata(self) -> Dict[str, Any]:
        with open(METADATA_FILE, "r") as f:
            return json.load(f)

    def _save_metadata(self, metadata: Dict[str, Any]):
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

    def store_record_encrypted(
        self, patient_id: str, record_payload: dict, device_private_key=None
    ) -> Tuple[str, str, str]:
        key = self.get_patient_key(patient_id)
        if key is None:
            raise ValueError("No key for patient. Generate one first.")
        f = Fernet(key)
        plaintext = json.dumps(record_payload, sort_keys=True).encode("utf-8")
        token = f.encrypt(plaintext)
        file_id = self.offchain.store_bytes(token)
        record_id = str(uuid.uuid4())
        fingerprint = sha256_hex_bytes(plaintext)

        signature_hex = None
        if device_private_key:
            signature = device_private_key.sign(
                plaintext,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256(),
            )
            signature_hex = signature.hex()

        metadata = self._load_metadata()
        metadata[record_id] = {
            "record_id": record_id,
            "patient_id": patient_id,
            "file_id": file_id,
            "uploaded_at": now_ts(),
            "fingerprint": fingerprint,
            "signature": signature_hex,
        }
        self._save_metadata(metadata)
        print(f"[EncryptedStore] Stored encrypted record {record_id}")
        return record_id, file_id, fingerprint

    def retrieve_record_decrypted(
        self, patient_id: str, record_id: str, device_public_key=None
    ) -> Optional[dict]:
        key = self.get_patient_key(patient_id)
        if key is None:
            raise ValueError("No key for patient.")
        metadata = self._load_metadata()
        if record_id not in metadata:
            return None
        file_id = metadata[record_id]["file_id"]
        token = self.offchain.retrieve_bytes(file_id)
        if token is None:
            return None
        f = Fernet(key)
        try:
            plaintext = f.decrypt(token)
            data = json.loads(plaintext.decode("utf-8"))
            if device_public_key and metadata[record_id]["signature"]:
                device_public_key.verify(
                    bytes.fromhex(metadata[record_id]["signature"]),
                    plaintext,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )
                print("[EncryptedStore] Signature verified: Data is authentic.")
            return data
        except Exception as e:
            print("[EncryptedStore] Decryption or verification error:", e)
            return None

# -------------------------
# Consent Smart Contract
# -------------------------
class ConsentSmartContract:
    def _init_(self, blockchain: SimpleBlockchain):
        self.bc = blockchain
        self.perm_store: Dict[str, Any] = {}

    def grant_access(
        self,
        patient_id: str,
        grantee_id: str,
        record_ids: Optional[List[str]] = None,
        expires_at: Optional[float] = None,
    ):
        perms = {
            "patient_id": patient_id,
            "grantee_id": grantee_id,
            "records": record_ids or "ALL",
            "expires_at": expires_at,
            "granted_at": now_ts(),
        }
        self.bc.add_block({"type": "GRANT", **perms})
        key = f"{patient_id}{grantee_id}"
        self.perm_store[key] = perms
        print(f"[ConsentSC] Granted {grantee_id} access to {patient_id}")

# -------------------------
# AI Analyzer
# -------------------------
class AIAnalyzer:
    def _init_(self, min_accuracy=0.80):
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self.min_accuracy = min_accuracy
        self.scaler: Optional[StandardScaler] = None

    def prepare_training_data(self):
        data = load_breast_cancer()
        return data.data, data.target, data.feature_names.tolist()

    def train(self):
        X, y, feature_names = self.prepare_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        self.scaler = StandardScaler().fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        model = RandomForestClassifier(
            n_estimators=400, max_depth=12, random_state=42, n_jobs=-1
        )
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        print("[AI] Accuracy:", acc)
        self.model = model
        self.feature_names = feature_names

    def analyze_record(self, record: dict) -> dict:
        result = {"summary": "", "risk_score": None, "highlights": {}}
        if not self.model or not self.scaler:
            return {"error": "Model not trained"}

        feature_vec = np.zeros(len(self.feature_names))
        for i, fname in enumerate(self.feature_names):
            if fname in record:
                feature_vec[i] = record[fname]

        feature_vec = self.scaler.transform([feature_vec])
        prob = self.model.predict_proba(feature_vec)[0][1]
        result["risk_score"] = float(prob)

        summary = f"Patient age {record.get('age','?')} - Malignancy Risk: {prob*100:.1f}%"
        result["summary"] = summary

        for fname in ["mean radius", "mean texture", "mean smoothness", "mean compactness"]:
            if fname in record:
                val = record[fname]
                if val > 0.15 * 100:
                    result["highlights"][fname] = f"{val} ⚠ Abnormal"
                else:
                    result["highlights"][fname] = f"{val} ✅ Normal"

        return result

# -------------------------
# Flask API
# -------------------------
app = Flask(_name_)

blockchain = SimpleBlockchain()
offchain = LocalOffChainStore()
enc_store = EncryptedDataStore(offchain)
consent_sc = ConsentSmartContract(blockchain)
ai = AIAnalyzer()
ai.train()

device_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
device_public_key = device_private_key.public_key()

@app.route("/submit_record", methods=["POST"])
def submit_record():
    data = request.json
    patient_id = data.get("patient_id")
    record = data.get("record")

    if not enc_store.get_patient_key(patient_id):
        enc_store.generate_patient_key(patient_id)

    rec_id, file_id, fp = enc_store.store_record_encrypted(
        patient_id, record, device_private_key
    )
    consent_sc.grant_access(patient_id, "doctor_demo_01")
    return jsonify({"record_id": rec_id, "file_id": file_id, "fingerprint": fp})

@app.route("/analyze_record/<record_id>", methods=["GET"])
def analyze_record(record_id):
    patient_id = request.args.get("patient_id")
    data = enc_store.retrieve_record_decrypted(patient_id, record_id, device_public_key)
    if not data:
        return jsonify({"error": "Record not found or invalid"}), 404
    analysis = ai.analyze_record(data)
    return jsonify({"record": data, "analysis": analysis})

# -------------------------
# Main
# -------------------------
if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)
