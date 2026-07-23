import streamlit as st
import pandas as pd
import asyncio
import json
import re
from datetime import datetime
from google.antigravity import Agent, LocalAgentConfig

# --- การตั้งค่าธีมและหน้าตาเว็บ (Modern & Professional) ---
st.set_page_config(
    page_title="สสจ.สระแก้ว | Smart Audit Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ระบบตรวจเช็ก Dependencies และโหมดการทำงาน (Online / Offline) ---
HAS_POSTGRES = False
try:
    import psycopg2
    HAS_POSTGRES = True
except ImportError:
    pass

HAS_PINECONE = False
try:
    from pinecone import Pinecone
    HAS_PINECONE = True
except ImportError:
    pass

HAS_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    pass

HAS_SUPABASE = False
try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    pass

# ตรวจสอบว่ามีข้อมูลการเชื่อมต่อใน st.secrets หรือไม่ (หลีกเลี่ยงข้อผิดพลาดหากยังไม่มีไฟล์ secrets.toml)
IS_ONLINE_MODE = False
USE_SUPABASE_API = False
USE_POSTGRES_DIRECT = False
admin_password_from_secrets = None

try:
    if "supabase" in st.secrets and HAS_SUPABASE:
        IS_ONLINE_MODE = True
        USE_SUPABASE_API = True
    elif "postgres" in st.secrets and HAS_POSTGRES:
        IS_ONLINE_MODE = True
        USE_POSTGRES_DIRECT = True
        
    admin_password_from_secrets = st.secrets.get("admin_password")
except Exception:
    pass

# ฟังก์ชันดึงไคลเอนต์ Supabase API
@st.cache_resource
def get_supabase_client():
    if not USE_SUPABASE_API:
        return None
    try:
        url = st.secrets["supabase"]["url"]
        # ลบ /rest/v1/ ออกหากผู้ใช้ใส่ URL เต็ม เพื่อรองรับ SDK
        if "/rest/v1/" in url:
            url = url.split("/rest/v1/")[0]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.sidebar.error(f"🔌 เชื่อมต่อ Supabase API ล้มเหลว: {e}")
        return None

# ฟังก์ชันดึงโมเดลแปลงเวกเตอร์ (เซฟแคชลงระบบเพื่อไม่โหลดใหม่ทุกรอบ)
@st.cache_resource
def load_embedding_model():
    if not HAS_TRANSFORMERS:
        return None
    try:
        # ใช้โมเดลภาษาไทย/หลายภาษาระดับความเร็วสูง
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        st.sidebar.warning(f"⚠️ โหลดโมเดลเวกเตอร์ไม่สำเร็จ: {e}")
        return None

# ฟังก์ชันเชื่อมต่อ PostgreSQL
def get_postgres_connection():
    if not HAS_POSTGRES:
        return None
    try:
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=int(st.secrets["postgres"].get("port", 5432)),
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            connect_timeout=3
        )
        return conn
    except Exception as e:
        st.sidebar.error(f"🔌 เชื่อมต่อ PostgreSQL ล้มเหลว: {e}")
        return None

# ฟังก์ชันตั้งค่าโครงสร้างตารางเริ่มต้นของ PostgreSQL/Supabase
def init_db():
    if not IS_ONLINE_MODE:
        return
        
    # 1. กรณีเชื่อมต่อผ่าน Supabase API Client
    if USE_SUPABASE_API:
        client = get_supabase_client()
        if client is None:
            return
        try:
            # ทดลองดึงตารางเพื่อทดสอบการมีอยู่
            client.table("audit_rules").select("id").limit(1).execute()
            st.session_state.has_pgvector = True  # Supabase คาดเดาว่าพร้อมสำหรับเวกเตอร์
        except Exception:
            st.sidebar.error("⚠️ ไม่พบตารางบน Supabase! กรุณาสร้างตารางใน SQL Editor บนเว็บ Supabase ก่อนใช้งาน")
            st.sidebar.info("💡 สามารถดูวิธีและคัดลอก SQL ได้ในคู่มือ [supabase_integration.md](file:///C:/Users/it-mo/.gemini/antigravity-ide/brain/c7d27caa-8905-4a96-a0c8-7c64ad6fba84/supabase_integration.md)")
        return
        
    # 2. กรณีเชื่อมต่อผ่าน PostgreSQL Direct TCP (psycopg2)
    if USE_POSTGRES_DIRECT:
        conn = get_postgres_connection()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                has_vector_ext = False
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    conn.commit()
                    has_vector_ext = True
                except Exception:
                    conn.rollback()
                    
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS audit_rules (
                        id SERIAL PRIMARY KEY,
                        rules_text TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                if has_vector_ext:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS audit_history (
                            id SERIAL PRIMARY KEY,
                            district VARCHAR(100) NOT NULL,
                            completeness_score INT,
                            accuracy_score INT,
                            document_score INT,
                            total_score INT,
                            discrepancies JSONB,
                            analysis_report TEXT,
                            embedding VECTOR(384),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                else:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS audit_history (
                            id SERIAL PRIMARY KEY,
                            district VARCHAR(100) NOT NULL,
                            completeness_score INT,
                            accuracy_score INT,
                            document_score INT,
                            total_score INT,
                            discrepancies JSONB,
                            analysis_report TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                conn.commit()
                st.session_state.has_pgvector = has_vector_ext
        except Exception as e:
            st.sidebar.error(f"🛠️ สร้างตารางเริ่มต้นล้มเหลว: {e}")
        finally:
            conn.close()

# ฟังก์ชันดึงเกณฑ์จาก PostgreSQL/Supabase
def load_audit_rules():
    default_rules = (
        "1. หัวข้อความครบถ้วน (10 คะแนน): ช่อง วันที่, รายการ, และ ยอดเงิน ห้ามเป็นค่าว่าง (ผิด 1 จุด หัก 1 คะแนน)\n"
        "2. หัวข้อตรรกะบัญชี (10 คะแนน): ยอดรวมฝั่งเดบิตและเครดิตต้องเท่ากัน (ถ้าไม่เท่ากัน ให้ 0 คะแนนในข้อนี้)\n"
        "3. หัวข้อความถูกต้องของเอกสาร (10 คะแนน): รายการที่มีคำว่า 'ถอนเงิน' ต้องมี เลขที่เอกสาร กำกับเสมอ"
    )
    if not IS_ONLINE_MODE:
        return default_rules
        
    # 1. กรณีเชื่อมต่อผ่าน Supabase API
    if USE_SUPABASE_API:
        client = get_supabase_client()
        if client is None:
            return default_rules
        try:
            res = client.table("audit_rules").select("rules_text").order("id", desc=True).limit(1).execute()
            if res.data:
                return res.data[0]["rules_text"]
            else:
                client.table("audit_rules").insert({"rules_text": default_rules}).execute()
                return default_rules
        except Exception as e:
            st.sidebar.warning(f"⚠️ ไม่สามารถเชื่อมดึงเกณฑ์จาก Supabase API (ใช้ค่า Default): {e}")
            return default_rules
            
    # 2. กรณีเชื่อมต่อผ่าน PostgreSQL Direct
    if USE_POSTGRES_DIRECT:
        conn = get_postgres_connection()
        if conn is None:
            return default_rules
        rules = default_rules
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT rules_text FROM audit_rules ORDER BY id DESC LIMIT 1;")
                row = cur.fetchone()
                if row:
                    rules = row[0]
                else:
                    cur.execute("INSERT INTO audit_rules (rules_text) VALUES (%s);", (default_rules,))
                    conn.commit()
        except Exception as e:
            st.sidebar.warning(f"⚠️ ไม่สามารถเชื่อมดึงเกณฑ์ (ใช้ระบบ Default): {e}")
        finally:
            conn.close()
        return rules
    return default_rules

# ฟังก์ชันเซฟเกณฑ์ลง PostgreSQL/Supabase
def save_audit_rules(rules_text):
    if not IS_ONLINE_MODE:
        st.session_state.audit_rules = rules_text
        st.success("💾 อัปเดตเกณฑ์การตรวจบัญชี (Local Mode) สำเร็จ!")
        return
        
    # 1. กรณีเชื่อมต่อผ่าน Supabase API
    if USE_SUPABASE_API:
        client = get_supabase_client()
        if client is None:
            st.error("🔌 บันทึกเกณฑ์ไม่สำเร็จ: ไม่พบไคลเอนต์ Supabase")
            return
        try:
            client.table("audit_rules").insert({"rules_text": rules_text}).execute()
            st.session_state.audit_rules = rules_text
            st.success("💾 อัปเดตเกณฑ์การตรวจบัญชีลง Supabase สำเร็จ!")
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดขณะเซฟค่าลง Supabase: {e}")
        return
        
    # 2. กรณีเชื่อมต่อผ่าน PostgreSQL Direct
    if USE_POSTGRES_DIRECT:
        conn = get_postgres_connection()
        if conn is None:
            st.error("🔌 บันทึกเกณฑ์ไม่สำเร็จ: การเชื่อมโยง PostgreSQL ล้มเหลว")
            return
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO audit_rules (rules_text) VALUES (%s);", (rules_text,))
                conn.commit()
            st.session_state.audit_rules = rules_text
            st.success("💾 อัปเดตเกณฑ์การตรวจบัญชีลง PostgreSQL สำเร็จ!")
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดขณะเซฟค่าลงฐานข้อมูล: {e}")
        finally:
            conn.close()

# ฟังก์ชันอัปเดตข้อมูลขึ้น Pinecone Cloud Index
def upsert_to_pinecone(audit_id, district, result_data):
    try:
        pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
        index = pc.Index(st.secrets["pinecone"]["index_name"])
        
        embedding_model = load_embedding_model()
        if embedding_model is None:
            return
        
        report_text = result_data.get("analysis_report", "")
        # สร้าง Vector
        vector = embedding_model.encode(report_text).tolist()
        
        # ส่งค่าขึ้น Pinecone
        index.upsert(
            vectors=[
                {
                    "id": f"audit_{audit_id}",
                    "values": vector,
                    "metadata": {
                        "audit_id": int(audit_id),
                        "district": district,
                        "total_score": int(result_data.get("total_score", 0)),
                        "discrepancies": json.dumps(result_data.get("discrepancies", [])),
                        "analysis_report": report_text,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
            ]
        )
    except Exception as e:
        st.warning(f"⚠️ บันทึกข้อมูลขึ้น Pinecone Vector Index ล้มเหลว: {e}")

# ฟังก์ชันเซฟประวัติคะแนนลง PostgreSQL/Supabase
def save_audit_result_to_db(district, result_data):
    if not IS_ONLINE_MODE:
        return None
    audit_id = None
    
    # 1. กรณีเชื่อมต่อผ่าน Supabase API
    if USE_SUPABASE_API:
        client = get_supabase_client()
        if client is not None:
            try:
                vector = None
                if HAS_TRANSFORMERS:
                    embedding_model = load_embedding_model()
                    if embedding_model is not None:
                        report_text = result_data.get("analysis_report", "")
                        vector = embedding_model.encode(report_text).tolist()
                
                payload = {
                    "district": district,
                    "completeness_score": int(result_data.get("completeness_score", 0)),
                    "accuracy_score": int(result_data.get("accuracy_score", 0)),
                    "document_score": int(result_data.get("document_score", 0)),
                    "total_score": int(result_data.get("total_score", 0)),
                    "discrepancies": result_data.get("discrepancies", []),
                    "analysis_report": result_data.get("analysis_report", "")
                }
                if vector is not None:
                    payload["embedding"] = vector # สามารถส่งเวกเตอร์แบบ list/array เข้า Supabase API ตรงๆ ได้เลย
                
                res = client.table("audit_history").insert(payload).execute()
                if res.data:
                    audit_id = res.data[0]["id"]
            except Exception as e:
                st.error(f"❌ ไม่สามารถเขียนประวัติลง Supabase API: {e}")
                
    # 2. กรณีเชื่อมต่อผ่าน PostgreSQL Direct (psycopg2)
    elif USE_POSTGRES_DIRECT:
        conn = get_postgres_connection()
        if conn is None:
            return None
        has_vector = st.session_state.get("has_pgvector", False)
        try:
            vector_str = None
            if has_vector and HAS_TRANSFORMERS:
                embedding_model = load_embedding_model()
                if embedding_model is not None:
                    report_text = result_data.get("analysis_report", "")
                    vector = embedding_model.encode(report_text).tolist()
                    vector_str = "[" + ",".join(map(str, vector)) + "]"
            
            with conn.cursor() as cur:
                if has_vector and vector_str is not None:
                    cur.execute("""
                        INSERT INTO audit_history 
                        (district, completeness_score, accuracy_score, document_score, total_score, discrepancies, analysis_report, embedding) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector) RETURNING id;
                    """, (
                        district,
                        int(result_data.get("completeness_score", 0)),
                        int(result_data.get("accuracy_score", 0)),
                        int(result_data.get("document_score", 0)),
                        int(result_data.get("total_score", 0)),
                        json.dumps(result_data.get("discrepancies", [])),
                        result_data.get("analysis_report", ""),
                        vector_str
                    ))
                else:
                    cur.execute("""
                        INSERT INTO audit_history 
                        (district, completeness_score, accuracy_score, document_score, total_score, discrepancies, analysis_report) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
                    """, (
                        district,
                        int(result_data.get("completeness_score", 0)),
                        int(result_data.get("accuracy_score", 0)),
                        int(result_data.get("document_score", 0)),
                        int(result_data.get("total_score", 0)),
                        json.dumps(result_data.get("discrepancies", [])),
                        result_data.get("analysis_report", "")
                    ))
                audit_id = cur.fetchone()[0]
                conn.commit()
        except Exception as e:
            st.error(f"❌ ไม่สามารถเขียนประวัติลงฐานข้อมูล: {e}")
        finally:
            conn.close()
            
    # ถ้ายังมีการตั้งค่า Pinecone เสริม ก็ยังคงอัปโหลดส่งไปคู่ขนานได้
    if audit_id is not None and "pinecone" in st.secrets:
        upsert_to_pinecone(audit_id, district, result_data)
        
    return audit_id

# ฟังก์ชันสืบค้นประวัติรายงานเชิงลึกจาก Supabase (pgvector) หรือ Pinecone
def semantic_search_reports(query_text, top_k=3):
    # 1. ค้นหาผ่าน Supabase API Client โดยเรียก RPC Function บนฐานข้อมูล
    if IS_ONLINE_MODE and USE_SUPABASE_API and HAS_TRANSFORMERS:
        client = get_supabase_client()
        if client is not None:
            try:
                embedding_model = load_embedding_model()
                if embedding_model is not None:
                    query_vector = embedding_model.encode(query_text).tolist()
                    
                    res = client.rpc("match_audit_history", {
                        "query_embedding": query_vector,
                        "match_threshold": 0.0,
                        "match_count": top_k
                    }).execute()
                    
                    matches = []
                    if res.data:
                        for row in res.data:
                            matches.append({
                                "id": f"audit_{row['id']}",
                                "score": float(row['similarity']) if row.get('similarity') is not None else 0.0,
                                "metadata": {
                                    "audit_id": row['id'],
                                    "district": row['district'],
                                    "total_score": row['total_score'],
                                    "discrepancies": json.dumps(row['discrepancies']) if isinstance(row['discrepancies'], (list, dict)) else row['discrepancies'],
                                    "analysis_report": row['analysis_report'],
                                    "created_at": row.get('created_at', '')
                                }
                            })
                        return matches
            except Exception as e:
                st.warning(f"⚠️ ค้นหาเวกเตอร์ผ่าน Supabase RPC ล้มเหลว (ลองใช้ Pinecone สำรอง): {e}")
                
    # 2. ค้นหาผ่าน PostgreSQL Direct pgvector (psycopg2)
    elif IS_ONLINE_MODE and USE_POSTGRES_DIRECT and st.session_state.get("has_pgvector", False) and HAS_TRANSFORMERS:
        conn = get_postgres_connection()
        if conn is not None:
            try:
                embedding_model = load_embedding_model()
                if embedding_model is not None:
                    query_vector = embedding_model.encode(query_text).tolist()
                    vector_str = "[" + ",".join(map(str, query_vector)) + "]"
                    
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT id, district, completeness_score, accuracy_score, document_score, total_score, discrepancies, analysis_report, created_at,
                                   1 - (embedding <=> %s::vector) AS similarity
                            FROM audit_history
                            WHERE embedding IS NOT NULL
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s;
                        """, (vector_str, vector_str, top_k))
                        
                        rows = cur.fetchall()
                        matches = []
                        for row in rows:
                            matches.append({
                                "id": f"audit_{row[0]}",
                                "score": float(row[9]) if row[9] is not None else 0.0,
                                "metadata": {
                                    "audit_id": row[0],
                                    "district": row[1],
                                    "total_score": row[5],
                                    "discrepancies": json.dumps(row[6]) if isinstance(row[6], (list, dict)) else row[6],
                                    "analysis_report": row[7],
                                    "created_at": row[8].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row[8], datetime) else str(row[8])
                                }
                            })
                        return matches
            except Exception as e:
                st.warning(f"⚠️ ค้นหาเวกเตอร์บน PostgreSQL ล้มเหลว (ลองใช้ Pinecone สำรอง): {e}")
            finally:
                conn.close()
                
    # 3. ค้นหาผ่าน Pinecone เป็นทางเลือกสำรอง
    if IS_ONLINE_MODE and "pinecone" in st.secrets and HAS_PINECONE:
        try:
            pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
            index = pc.Index(st.secrets["pinecone"]["index_name"])
            embedding_model = load_embedding_model()
            if embedding_model is None:
                return []
            
            query_vector = embedding_model.encode(query_text).tolist()
            response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            return response.get("matches", [])
        except Exception as e:
            st.error(f"❌ สืบค้นความหมายด้วย Pinecone ล้มเหลว: {e}")
            return []
            
    return []

# --- ระบบจัดตั้งเกณฑ์การตรวจและค้างสถานะ ---
if "has_pgvector" not in st.session_state:
    st.session_state.has_pgvector = False

if "audit_rules" not in st.session_state:
    st.session_state.audit_rules = load_audit_rules()

if "audit_result" not in st.session_state:
    st.session_state.audit_result = None
if "last_analyzed_district" not in st.session_state:
    st.session_state.last_analyzed_district = None
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None
if "last_selected_district" not in st.session_state:
    st.session_state.last_selected_district = None

# --- เริ่มทำงานจัดตั้งโครงสร้างเมื่อแอปพลิเคชันทำงาน ---
if IS_ONLINE_MODE:
    init_db()

# --- ส่วนควบคุมการแสดงหน้าจอ (Sidebar) ---
with st.sidebar:
    # จัดโลโก้ให้อยู่ตรงกลางและมีขนาดใหญ่ขึ้น 2 เท่า (ใช้สัดส่วนคอลัมน์กึ่งกลาง)
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        st.image("moph_logo.png", use_container_width=True)
    
    # จัดข้อความแผงควบคุมให้อยู่ตรงกลางสอดคล้องกับโลโก้
    st.markdown("<h2 style='text-align: center; margin-top: 0px; margin-bottom: 0px;'>แผงควบคุมระบบ</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 0.85rem; margin-top: 5px; margin-bottom: 20px;'>ระบบตรวจบัญชีอัตโนมัติ สสจ.สระแก้ว v2.0</p>", unsafe_allow_html=True)
    
    # แจ้งเตือนสถานะการเชื่อมต่อฐานข้อมูล
    st.markdown("---")
    if IS_ONLINE_MODE:
        if USE_SUPABASE_API:
            db_type = "Supabase API"
            vector_status = " + pgvector"
        else:
            db_host = st.secrets.get("postgres", {}).get("host", "")
            db_type = "Supabase" if "supabase" in db_host else "PostgreSQL"
            vector_status = " + pgvector" if st.session_state.get("has_pgvector", False) else ""
            
        pinecone_status = " + Pinecone" if "pinecone" in st.secrets else ""
        st.success(f"🟢 สถานะ: เชื่อมต่อ {db_type}{vector_status}{pinecone_status} (Online)")
    else:
        st.warning("⚠️ สถานะ: โหมดใช้งานชั่วคราว (Offline Mode)")
        st.caption("แอปพลิเคชันเปิดทำงานได้ปกติ แต่จะไม่เซฟประวัติลงฐานข้อมูลและไม่สามารถค้นหาอัจฉริยะได้ (ตรวจสอบ secrets.toml หรือไลบรารีขาดหาย)")
        if not (HAS_POSTGRES and HAS_PINECONE and HAS_TRANSFORMERS):
            st.info("กรุณารันติดตั้งคำสั่งเพื่อรองรับระบบฐานข้อมูล:\n`pip install -r requirements.txt` เพื่อความสมบูรณ์แบบ")
            
    st.markdown("---")
    
    # ⚙️ หน้าต่างสำหรับผู้ดูแลระบบ (Admin) เพื่อเพิ่ม/แก้ไข เงื่อนไขตรวจบัญชีแบบ Dynamic
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    st.subheader("🛠️ ตั้งค่าเกณฑ์การตรวจสอบโดย AI")
    
    if not st.session_state.is_admin:
        admin_pwd_input = st.text_input("ระบุรหัสผ่านผู้ดูแลระบบ (Admin):", type="password", key="admin_pwd_widget")
        if st.button("🔓 เข้าสู่ระบบแอดมิน", use_container_width=True):
            correct_pwd = admin_password_from_secrets if admin_password_from_secrets is not None else "admin1234"
            if admin_pwd_input == correct_pwd:
                st.session_state.is_admin = True
                st.success("🔓 ล็อกอินสำเร็จ!")
                st.rerun()
            else:
                st.error("🔑 รหัสผ่านไม่ถูกต้อง")
    else:
        custom_rules = st.text_area(
            "ระบุเงื่อนไขและคะแนนที่ต้องการให้ AI ตรวจจับ (ปรับแต่งได้ตลอดเวลา):",
            value=st.session_state.audit_rules,
            height=250,
            help="คุณสามารถเพิ่มเงื่อนไขใหม่ๆ เช่น ตรวจสอบงบจำเพาะ หรือการลงบัญชีผิดประเภท ได้ที่นี่"
        )
        if st.button("💾 บันทึกและปรับปรุงเงื่อนไข", use_container_width=True):
            save_audit_rules(custom_rules)
            
        st.markdown("---")
        if st.button("🔒 ออกจากระบบผู้ดูแลระบบ", type="secondary", use_container_width=True):
            st.session_state.is_admin = False
            st.rerun()

# --- หน้าจอหลักสำหรับผู้ใช้งาน (Main Dashboard) ---
st.markdown("<h1 style='color: #1E3A8A; margin-bottom: 0px;'>🩺 Smart Audit Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6B7280; font-size: 1.1rem;'>ระบบวิเคราะห์และตรวจสอบรายการบัญชีสำนักงานสาธารณสุขอำเภอ จังหวัดสระแก้ว</p>", unsafe_allow_html=True)
st.markdown("---")

# การจัด Layout การรับข้อมูลให้เป็นสัดส่วนสวยงาม
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 📥 ส่วนส่งข้อมูลตรวจสอบ")
    
    # เลือกอำเภอในจังหวัดสระแก้ว
    district = st.selectbox(
        "เลือกสำนักงานสาธารณสุขอำเภอ (สสอ.):",
        ["สสอ.เมืองสระแก้ว", "สสอ.คลองหาด", "สสอ.ตาพระยา", "สสอ.วังน้ำเย็น", "สสอ.วัฒนานคร", "สสอ.อรัญประเทศ", "สสอ.เขาฉกรรจ์", "สสอ.โคกสูง", "สสอ.วังสมบูรณ์"]
    )
    
    # กล่องลากวางไฟล์ Excel ที่ทันสมัย
    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์รายงานบัญชีประจำเดือน (.xlsx)", 
        type=["xlsx"],
        help="กรุณาตรวจสอบว่าหัวตารางโครงสร้างคอลัมน์ถูกต้องตามมาตรฐาน"
    )
    
    execute_button = st.button("🚀 เริ่มกระบวนการวิเคราะห์ด้วย AI", use_container_width=True, type="primary")

# ตรวจสอบการเปลี่ยนไฟล์หรือเปลี่ยน สสอ. เพื่อรีเซ็ตผลลัพธ์เก่า ป้องกันแสดงผลข้ามอำเภอ
current_file_name = uploaded_file.name if uploaded_file is not None else None
if (current_file_name != st.session_state.last_file_name or 
    district != st.session_state.last_selected_district):
    st.session_state.audit_result = None
    st.session_state.last_analyzed_district = None
    st.session_state.last_file_name = current_file_name
    st.session_state.last_selected_district = district

df = None

with col2:
    # แยกพื้นที่ทำงานเป็น 2 แท็บ (แท็บวิเคราะห์หลัก และ แท็บสืบค้นอัจฉริยะ)
    tab1, tab2 = st.tabs(["📋 การวิเคราะห์หลัก", "🔍 ค้นหาประวัติวิเคราะห์ด้วย AI"])
    
    with tab1:
        st.markdown("### 📋 ข้อมูลพรีวิวจากไฟล์")
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                
                # การแจ้งเตือนข้อจำกัดของจำนวนแถว (Token Safety)
                row_count = len(df)
                if row_count > 1000:
                    st.warning(f"⚠️ คำเตือน: ไฟล์มีข้อมูล {row_count:,} รายการ ซึ่งมีขนาดค่อนข้างใหญ่ (แนะนำไม่เกิน 1,000 รายการเพื่อให้ AI ตรวจสอบได้อย่างแม่นยำที่สุด)")
                else:
                    st.caption(f"📊 ตรวจพบข้อมูลทั้งหมด {row_count} รายการ")
                    
                st.dataframe(df, use_container_width=True, height=220)
                
                # ระบบตรวจคอลัมน์เบื้องต้น (Column Pre-validation)
                required_cols = ["วันที่", "รายการ", "ยอดเงิน", "เดบิต", "เครดิต", "เลขที่เอกสาร"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"⚠️ คำเตือน: โครงสร้างตารางของท่านขาดคอลัมน์สำคัญ: {', '.join(missing_cols)} ซึ่งอาจส่งผลต่อการคำนวณคะแนนของ AI")
                else:
                    st.success("✅ โครงสร้างตารางสอดคล้องตามคอลัมน์มาตรฐานสำหรับการตรวจบัญชี")
                    
            except Exception as e:
                st.error(f"ไม่สามารถอ่านไฟล์ Excel ได้: {e}")
        else:
            st.info("💡 กรุณาอัปโหลดไฟล์ Excel ฝั่งซ้ายมือเพื่อดูพรีวิวข้อมูลบัญชี")

    with tab2:
        st.markdown("### 🔍 ค้นหาบทวิเคราะห์อัจฉริยะเชิงความหมาย (Semantic Search)")
        st.write("พิมพ์สอบถามหัวข้อหรือคำค้นภาษาไทย เช่น *'สสอ.ที่มีปัญหาเดบิตกับเครดิตไม่เท่ากัน'* หรือ *'ตรวจสอบความผิดพลาดด้านเอกสารหาย'* เพื่อดึงข้อมูลรายงานเก่าจากคลัง Pinecone ขึ้นมาเปรียบเทียบ")
        
        if not IS_ONLINE_MODE:
            st.info("💡 บริการสืบค้นความหมายอัจฉริยะจะเปิดให้ใช้งานเมื่อตั้งค่ารหัสเชื่อมโยง Pinecone เรียบร้อยในไฟล์ secrets.toml")
        else:
            query = st.text_input("ป้อนเรื่องที่ต้องการสืบค้นประวัติ:")
            search_btn = st.button("🔎 เริ่มการค้นหาความหมาย", use_container_width=True)
            if search_btn and query:
                with st.spinner("🔄 กำลังดึงประวัติรายงานที่เกี่ยวข้องจาก Pinecone..."):
                    matches = semantic_search_reports(query)
                    if not matches:
                        st.info("ไม่พบประวัติผลลัพธ์ที่สอดคล้องกับเนื้อหาที่ค้นหา")
                    else:
                        st.success(f"พบรายงานเก่าที่มีความคล้ายคลึงมากที่สุด {len(matches)} รายการ:")
                        for match in matches:
                            meta = match.get("metadata", {})
                            score = match.get("score", 0)
                            
                            with st.expander(f"📌 {meta.get('district', 'สสอ. ไม่ระบุ')} | คะแนนรวม: {meta.get('total_score', 0)} (ความคล้ายคลึง: {score:.2%})"):
                                st.write(f"📅 **บันทึกข้อมูลเมื่อ:** {meta.get('created_at', 'ไม่ระบุ')}")
                                st.markdown("**📝 รายงานบทวิเคราะห์เชิงลึก:**")
                                st.info(meta.get("analysis_report", ""))
                                
                                errors_str = meta.get("discrepancies", "[]")
                                try:
                                    errors_list = json.loads(errors_str)
                                except Exception:
                                    errors_list = []
                                if errors_list:
                                    st.markdown("**❌ รายการจุดบกพร่องที่พบ:**")
                                    for err in errors_list:
                                        st.error(err)

# --- ส่วนหลังบ้านที่ประมวลผลร่วมกับ Antigravity SDK ---
if uploaded_file is not None and df is not None and execute_button:
    # การย้ายผลการแสดงผลลัพธ์หลักมาเขียนใต้บรรทัดแบ่ง
    st.markdown("---")
    st.markdown("### 🧠 ผลการตรวจสอบและประเมินผลโดย AI")
    
    with st.spinner("🔄 Antigravity Agent กำลังประมวลผลข้อมูลและคำนวณคะแนนตามเกณฑ์แบบเรียลไทม์..."):
        
        # 1. แปลง Dataframe จาก Excel เป็น string เพื่อส่งให้ AI (แก้ปัญหาค่าว่างล่วงหน้า)
        excel_data_str = df.fillna("").to_csv(index=False)
        
        # 2. ปรับแต่งโครงสร้างคำสั่งหลังบ้าน (System Prompt) ให้นำ 'เงื่อนไขที่ผู้ใช้กำหนด' ไปรัน
        SYSTEM_INSTRUCTION = f"""
        คุณคือ AI นักตรวจบัญชีระดับสูง ทำหน้าที่ตรวจไฟล์บัญชีของสาธารณสุขอำเภอในจังหวัดสระแก้ว
        จงนำข้อมูลบัญชีที่ผู้ใช้ส่งมา นำมาตรวจสอบวิเคราะห์อย่างละเอียดตาม "เกณฑ์ที่ผู้กำหนดด้านล่างนี้" อย่างเคร่งครัด
        
        [เกณฑ์การตรวจสอบและให้คะแนนที่ต้องปฏิบัติตาม]
        {st.session_state.audit_rules}
        
        [ข้อบังคับในการส่งผลลัพธ์]
        คุณต้องวิเคราะห์และคำนวณคะแนนออกมา แล้วตอบกลับมาในรูปแบบรูปแบบ JSON String ที่ถูกต้องตามโครงสร้างนี้เท่านั้น (ห้ามมีข้อความอื่นนอกเหนือจาก JSON):
        {{
          "completeness_score": คะแนนหัวข้อที่ 1 (ตัวเลขเต็ม 10),
          "accuracy_score": คะแนนหัวข้อที่ 2 (ตัวเลขเต็ม 10),
          "document_score": คะแนนหัวข้อที่ 3 (ตัวเลขเต็ม 10),
          "total_score": คะแนนรวมทั้งหมดที่คุณคำนวณได้,
          "discrepancies": ["รายการจุดที่ผิดพลาดข้อที่ 1 (ระบุแถวหรือรายการที่ผิดด้วย)", "จุดที่ผิดข้อที่ 2"],
          "analysis_report": "สรุปผลวิเคราะห์เชิงลึกและคำแนะนำแก่สาธารณสุขอำเภอนี้"
        }}
        """
        
        # 3. ฟังก์ชัน Asynchronous สำหรับเรียกใช้ Antigravity SDK
        async def process_audit():
            config = LocalAgentConfig(
                system_instructions=SYSTEM_INSTRUCTION,
                temperature=0.1  # บังคับค่าความเสถียรของตรรกะคณิตศาสตร์และบัญชี
            )
            async with Agent(config) as agent:
                prompt_payload = f"ข้อมูลบัญชีประจำเดือนของ สสอ.: {district}\n\n[Data]\n{excel_data_str}"
                response = await agent.run(prompt_payload)
                return response.text

        # 4. สั่งรันและรับผลลัพธ์
        try:
            raw_response = asyncio.run(process_audit())
            
            # ค้นหารูปแบบ JSON ในเครื่องหมายปีกกาด้วย Regex เพื่อป้องกันล่มจากข้อความเกริ่นของ AI
            json_match = re.search(r"(\{.*\})", raw_response, re.DOTALL)
            if json_match:
                clean_json = json_match.group(1)
                result_data = json.loads(clean_json)
                
                # บันทึกข้อมูลผลลัพธ์ลง Session State
                st.session_state.audit_result = result_data
                st.session_state.last_analyzed_district = district
                
                # เซฟลง PostgreSQL ฐานข้อมูล และ Pinecone คอนทูแอร์เวกเตอร์อัตโนมัติ
                save_audit_result_to_db(district, result_data)
            else:
                raise ValueError("ไม่พบรูปแบบโครงสร้างข้อมูล JSON ที่สมบูรณ์ในคำตอบของระบบ AI")
            
        except Exception as ex:
            st.error(f"เกิดข้อผิดพลาดในระบบการเชื่อมต่อหรือการวิเคราะห์ข้อมูล: {ex}")
            st.session_state.audit_result = None

# --- ส่วนดึงการแสดงผลลัพธ์จาก Session State (ทำให้อยู่คงทนเมื่อเปลี่ยนองค์ประกอบอื่นๆ ใน UI) ---
if st.session_state.audit_result is not None:
    st.markdown("---")
    st.markdown(f"### 🧠 ผลการตรวจสอบและประเมินผลโดย AI: {st.session_state.last_analyzed_district}")
    
    result_data = st.session_state.audit_result
    
    # --- ส่วนแสดงการวัดผล (Metrics & Score Dashboard) ---
    st.success(f"🎉 การตรวจสอบ {st.session_state.last_analyzed_district} เสร็จสิ้นสมบูรณ์!")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("คะแนนความครบถ้วน", f"{result_data.get('completeness_score', 0)} / 10")
    m2.metric("คะแนนตรรกะบัญชี", f"{result_data.get('accuracy_score', 0)} / 10")
    m3.metric("คะแนนความถูกต้องเอกสาร", f"{result_data.get('document_score', 0)} / 10")
    m4.metric("🏆 คะแนนรวมสุทธิ", f"{result_data.get('total_score', 0)} คะแนน", delta=None)
    
    # --- ส่วนแสดงจุดบกพร่องและรายงานสรุป ---
    st.markdown("#### 🔍 รายการข้อผิดพลาดที่ตรวจพบ (Discrepancies)")
    errors = result_data.get("discrepancies", [])
    if len(errors) == 0:
        st.info("🟩 ไม่พบข้อผิดพลาดในไฟล์บัญชีนี้ ข้อมูลถูกต้องตามเกณฑ์")
    else:
        for error in errors:
            st.error(f"❌ {error}")
    
    st.markdown("#### 📝 รายงานการวิเคราะห์และข้อเสนอแนะภาพรวม")
    st.info(result_data.get("analysis_report", "ไม่มีข้อมูลรายงาน"))
    
    # --- ส่วนของการส่งออกรายงาน (Export Report) ---
    report_txt = f"""==================================================
รายงานผลการตรวจบัญชีของ {st.session_state.last_analyzed_district}
==================================================
คะแนนสุทธิรวม: {result_data.get('total_score', 0)} คะแนน
- คะแนนความครบถ้วน: {result_data.get('completeness_score', 0)}/10
- คะแนนตรรกะบัญชี: {result_data.get('accuracy_score', 0)}/10
- คะแนนความถูกต้องเอกสาร: {result_data.get('document_score', 0)}/10

รายการจุดบกพร่องที่ตรวจสอบพบ:
"""
    if len(errors) == 0:
        report_txt += "- ไม่พบจุดบกพร่อง\n"
    else:
        for error in errors:
            report_txt += f"- [พบข้อผิดพลาด] {error}\n"
            
    report_txt += f"\nรายละเอียดบทวิเคราะห์และข้อเสนอแนะ:\n{result_data.get('analysis_report', '')}\n"
    
    st.download_button(
        label="📥 ดาวน์โหลดรายงานผลวิเคราะห์ฉบับสมบูรณ์ (TXT)",
        data=report_txt,
        file_name=f"audit_report_{st.session_state.last_analyzed_district}.txt",
        mime="text/plain",
        use_container_width=True
    )
