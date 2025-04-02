import os
import json
import pdfplumber
from pyvis.network import Network

# ======= 設定（必要に応じて変更可能） =======
PDF_FILE_PATH = r"AdditionalDocuments"  # ファイル単位でもフォルダ単位でも指定可能
PAGE_START = 6    # 読み取り開始ページ（1-based）
PAGE_END = 8      # 読み取り終了ページ（1-based, 終了ページも含む）
CHUNK_SIZE = 400  # テキストチャンクのサイズ
CHUNK_OVERLAP = 100  # チャンク間の重複部分
OUTPUT_KNOWLEDGE_DB_FILE = "knowledge_db.json"  # ナレッジDBの保存先
OUTPUT_GRAPH_HTML = "graph.html"  # グラフ可視化結果のHTML出力ファイル
DOC_ID = "doc_1"  # 単一ファイルの場合のデフォルトドキュメントID
# =============================================

# --- 技術知識抽出用のプロンプトテンプレート ---
PROMPT_TEMPLATE = (
    "以下のテキストから、技術的な手法、プロセス、パラメータ、用語、及びそれらの相互関係を抽出してください。"
    "返答は必ず以下のJSON形式のみとし、余計なテキストや説明文を含めず、厳密に出力してください。\n\n"
    "【出力例】:\n"
    "```\n"
    "{{\n"
    "  \"技術的な手法\": [\"手法1\", \"手法2\"],\n"
    "  \"プロセス\": [\"プロセス1\", \"プロセス2\"],\n"
    "  \"パラメータ\": [\"パラメータ1\", \"パラメータ2\"],\n"
    "  \"用語\": [\"用語1\", \"用語2\", \"用語3\", \"用語4\", \"用語5\"],\n"
    "  \"関係性\": [\n"
    "    {{\"source\": \"手法1\", \"target\": \"プロセス1\", \"relation\": \"関連性の種類\"}},\n"
    "    {{\"source\": \"パラメータ1\", \"target\": \"用語3\", \"relation\": \"関連性の種類\"}}\n"
    "  ]\n"
    "}}\n"
    "```\n\n"
    "テキスト:\n\n{chunk}"
)


#####################################################################################################
#####################################################################################################

# --- PDFからテキストを抽出 ---
def extract_text_from_pdf(pdf_path, start_page, end_page):
    """
    指定したPDFファイルから、start_page～end_page（1-based）のテキストを抽出して連結した文字列を返す
    """
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        # 1-basedのページ番号を0-indexに変換して抽出
        for page in pdf.pages[start_page - 1:end_page]:
            texts.append(page.extract_text())
    return "\n".join(texts)

# --- テキストをチャンク分割 ---
def split_text(text, chunk_size, chunk_overlap):
    """
    テキストを指定サイズのチャンクに分割する（チャンク間は指定分重複）
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

# --- OpenAI API呼び出し ---
def call_openai_api(chunks):
    """
    各チャンクに対してOpenAI APIを呼び出し、抽出した技術知識のレスポンスをリストで返す
    """
    responses = []
    for i, chunk in enumerate(chunks):
        prompt = PROMPT_TEMPLATE.format(chunk=chunk)
        print(f"Processing chunk {i+1}:\n{chunk}\n")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-us-MEM-DX-openai-001",
                messages=[
                    {"role": "system", "content": "あなたは技術知識抽出の専門家です。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.2
            )
            extracted = response.choices[0].message.content.strip()
            print("Response:")
            print(extracted)
            responses.append(extracted)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
    return responses

# --- レスポンスのパース ---
def parse_response(resp_str):
    """
    APIレスポンスからマークダウン部分（```json ... ```）を除去し、JSONオブジェクトとして返す
    """
    lines = resp_str.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    json_str = "\n".join(lines)
    return json.loads(json_str)

# --- ナレッジDBの読み込み・初期化 ---
def load_knowledge_db(db_file):
    """
    ナレッジDBのJSONファイルを読み込む。存在しない場合は初期構造を返す。
    """
    if os.path.exists(db_file):
        with open(db_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"documents": [], "graph": {"nodes": [], "edges": []}}

def save_knowledge_db(db, db_file):
    """
    ナレッジDBをJSONファイルに保存する
    """
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

# --- ナレッジDBの更新 ---〈修正する部分〉 update_knowledge_db を既存のドキュメントがある場合は上書き更新し、対応するグラフエッジを一旦削除してから再追加するように修正

def update_knowledge_db(db, extracted_knowledge, full_text, doc_id):
    doc_id_str = str(doc_id)
    # 既存のドキュメントがあるかチェックして更新（なければ追加）
    doc_exists = False
    for i, doc in enumerate(db["documents"]):
        if doc.get("id") == doc_id_str:
            db["documents"][i]["full_text"] = full_text
            db["documents"][i]["extracted_knowledge"] = extracted_knowledge
            doc_exists = True
            break
    if not doc_exists:
        db["documents"].append({
            "id": doc_id_str,
            "full_text": full_text,
            "extracted_knowledge": extracted_knowledge
        })
    
    # ドキュメントノードの追加（既にあれば何もしない）
    if not any(node["id"] == doc_id_str for node in db["graph"]["nodes"]):
        db["graph"]["nodes"].append({
            "id": doc_id_str,
            "label": doc_id_str,
            "group": "document"
        })
    
    # 既存のドキュメントに紐づくエッジを一旦削除（更新するため）
    db["graph"]["edges"] = [edge for edge in db["graph"]["edges"] if edge["source"] != doc_id_str]
    
    # カテゴリ毎のノード・エッジの追加
    categories = {
        "技術的な手法": "technique",
        "プロセス": "process",
        "パラメータ": "parameter",
        "用語": "term"
    }
    for key, group in categories.items():
        items = extracted_knowledge.get(key, [])
        for item in items:
            item_str = str(item)
            if not any(node["id"] == item_str for node in db["graph"]["nodes"]):
                db["graph"]["nodes"].append({
                    "id": item_str,
                    "label": item_str,
                    "group": group
                })
            db["graph"]["edges"].append({
                "source": doc_id_str,
                "target": item_str,
                "label": key
            })
    
    # 関係性の追加（各関係を示すオブジェクトのリスト）
    relationships = extracted_knowledge.get("関係性", [])
    if isinstance(relationships, list):
        for rel in relationships:
            src = rel.get("source")
            target = rel.get("target")
            relation_type = rel.get("relation", "関連")
            if src and target:
                src_str = str(src)
                target_str = str(target)
                if not any(node["id"] == src_str for node in db["graph"]["nodes"]):
                    db["graph"]["nodes"].append({
                        "id": src_str,
                        "label": src_str,
                        "group": "relation"
                    })
                if not any(node["id"] == target_str for node in db["graph"]["nodes"]):
                    db["graph"]["nodes"].append({
                        "id": target_str,
                        "label": target_str,
                        "group": "relation"
                    })
                db["graph"]["edges"].append({
                    "source": src_str,
                    "target": target_str,
                    "label": relation_type
                })
    return db

# --- グラフの可視化 ---
def visualize_graph(db, output_filename):
    """
    ナレッジDB内のグラフ情報をpyvisで可視化し、HTMLファイルとして出力する
    """
    net = Network(height="600px", width="100%", directed=False, notebook=False)
    for node in db["graph"]["nodes"]:
        net.add_node(node["id"], label=node["label"], group=node["group"])
    for edge in db["graph"]["edges"]:
        net.add_edge(edge["source"], edge["target"], label=edge["label"])
    html = net.generate_html()
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"グラフが '{output_filename}' として保存されました。")

#####################################################################################################
#####################################################################################################

# 〈追加する部分〉
# main() 内で、既存のナレッジDBを一度だけ読み込んでから全PDFファイル分をループ処理し、更新後に保存＆グラフを再生成するように修正

def main():
    # PDF_FILE_PATH がディレクトリの場合は、フォルダ内のPDFファイル一覧を取得
    if os.path.isdir(PDF_FILE_PATH):
        pdf_files = [
            os.path.join(PDF_FILE_PATH, f)
            for f in os.listdir(PDF_FILE_PATH)
            if f.lower().endswith('.pdf')
        ]
    else:
        pdf_files = [PDF_FILE_PATH]
    
    # 既存のナレッジDBを1回だけ読み込む
    db = load_knowledge_db(OUTPUT_KNOWLEDGE_DB_FILE)
    
    # 各PDFファイルに対して知識抽出とDB更新を実行
    for pdf_path in pdf_files:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Processing PDF file: {pdf_path}")
        
        # 1. PDFからテキスト抽出
        full_text = extract_text_from_pdf(pdf_path, PAGE_START, PAGE_END)
        print("PDFからテキストを抽出しました。")
        
        # 2. テキストをチャンク分割
        chunks = split_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"テキストを {len(chunks)} 個のチャンクに分割しました。")
        
        # 3. 各チャンクをOpenAI APIに渡して技術知識抽出
        api_responses = call_openai_api(chunks)
        print("APIからの応答を受信しました。")
        
        # 4. レスポンスをパースし統合
        extracted_knowledge_all = {}
        for response in api_responses:
            try:
                parsed = parse_response(response)
                for key, value in parsed.items():
                    if key in extracted_knowledge_all:
                        if isinstance(extracted_knowledge_all[key], list) and isinstance(value, list):
                            extracted_knowledge_all[key].extend(value)
                        elif isinstance(extracted_knowledge_all[key], dict) and isinstance(value, dict):
                            extracted_knowledge_all[key].update(value)
                    else:
                        extracted_knowledge_all[key] = value
            except Exception as e:
                print(f"レスポンスのパース中にエラーが発生しました: {e}")
        print("抽出された技術知識（統合結果）:")
        print(json.dumps(extracted_knowledge_all, ensure_ascii=False, indent=2))
        
        # 既存DBに対して新たなナレッジを追加（または更新）
        db = update_knowledge_db(db, extracted_knowledge_all, full_text, doc_id)
    
    # すべてのPDF処理後にDBを保存＆グラフを可視化
    save_knowledge_db(db, OUTPUT_KNOWLEDGE_DB_FILE)
    print(f"ナレッジDBが '{OUTPUT_KNOWLEDGE_DB_FILE}' に保存されました。")
    visualize_graph(db, OUTPUT_GRAPH_HTML)

if __name__ == "__main__":
    main()
