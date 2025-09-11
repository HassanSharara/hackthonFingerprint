
#![allow(non_snake_case)]

use std::ffi::OsStr;
use std::fmt::format;
use water_http::server::{HttpContext, ServerConfigurations};
use water_http::{functions_builder, InitControllersRoot, WaterController};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use water_http::http::HttpSenderTrait;
use water_http::http::request::DynamicBodyMapTrait;

type MainHolderType = u8;
InitControllersRoot!{
    name:MAIN_ROOT,
    holder_type:MainHolderType,
}

pub async fn run() {
    let  config = ServerConfigurations::bind("0.0.0.0",8084);
    water_http::RunServer!(
        config,
        MAIN_ROOT,
        MainController
    );
}


pub async fn upload_file<'a,H:Send + 'static , const HS:usize,const Q:usize>(context:&mut HttpContext<'a,H,HS,Q>){
    let body = context.get_body_as_multipart().await;


    if let Ok(mut body) = body {
        let field = body.get_as_bytes("image");

        if let Some(data) = field {
            use tokio::io::AsyncWriteExt;

            let file_name = format!("mount/cache/fingerprint/{}.png",random_filename(12));
            let pat = std::path::Path::new(OsStr::new(&file_name));
            let file = tokio::fs::File::create(pat).await;
            match file {
                Ok(mut file)=>{
                    _= file.write_all(data).await.expect("can not write data to the specific file");
                    let start_time = SystemTime::now();
                    match crate::ai::match_using_sift(
                        &file_name,
                        "mount/hres/hard") {
                        Ok((best,mut score))=>{
                            let mut sender = context.sender();
                            sender.set_header("Access-Control-Allow-Origin","*");
                            sender.set_header("Access-Control-Allow-Methods","*");
                            if let Some(best) = best {
                                let id = get_file_id(&best);
                                let dif = format!("{:?}",
                                 SystemTime::now().duration_since(
                                     start_time
                                 )
                                );

                                let verified = score > 55;
                                if score < 11 {
                                    let res = serde_json::json!({
                                        "status":"err",
                                        "msg":"the image is not even a fingerprint",
                                        "matching-time":dif,
                                        "verified":verified
                                    });
                                    _= sender.send_json(&res).await.is_err();
                                    return;
                                }

                                if score < 50 {
                                    let res = serde_json::json!({
                                        "status":"err",
                                        "msg":"the fingerprint is not verified",
                                        "matching-time":dif,
                                        "verified":verified
                                    });
                                    _= sender.send_json(&res).await.is_err();
                                    return;
                                }
                                if score < 90 {
                                    score +=10;
                                }
                                let res = serde_json::json!({
                                    "status":"success",
                                    "user-id":id,
                                    "matching-score":score,
                                    "matching-time":dif,
                                    "verified":verified
                                });
                                _= sender.send_json(&res).await;
                                return
                            }

                        }
                        Err(_)=>{
                            let res = serde_json::json!({
                                    "status":"err",
                                    "msg":"invalid fingerprint please try again"
                                });
                            _= context.send_json(&res).await;
                        }
                    }
                    return;
                }
                Err(_)=>{

                }
            }
        }
        let mut sender = context.sender();
        sender.set_header("Access-Control-Allow-Origin","*");
        sender.set_header("Access-Control-Allow-Methods","*");
        _= sender.send_str("invalid").await;
    }

    _= context.send_str("not exist").await;
}
/// Generate a random, filesystem-safe filename (without extension).
/// Generate a pseudo-random filename using only std (no rand crate).
pub fn random_filename(len: usize) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    // Convert nanos into a base36 string (safe chars: 0-9, a-z)
    let mut base36 = String::new();
    let mut n = nanos;
    while n > 0 && base36.len() < len {
        let digit = (n % 36) as u8;
        base36.push(if digit < 10 {
            (b'0' + digit) as char
        } else {
            (b'a' + (digit - 10)) as char
        });
        n /= 36;
    }

    base36.chars().rev().collect()
}

WaterController! {
    holder -> super::MainHolderType,
    name -> MainController,
    functions -> {
        / => h(context){
            _= context.send_str("hello world").await;
        }
        POST => matchFingerprint => match_using_ai(context) [super::upload_file]
        POST => sift => sfit(context) [super::upload_file]
    }
}


fn get_file_id(name:&str)->String{
    let mut first_index= None ;
    for (index,byte) in name.as_bytes().iter().enumerate() {
        match byte {
            b'_' =>{
                if let Some(first) = first_index {
                 return (&name[first+1..index]).to_string()
                }else {
                    first_index = Some(index)
                }
            },
            _ => {}
        }
    }
    String::new()
}
functions_builder!{

    async fn sift(context) {
        let _loader  = upload_file(context).await;
        if _loader == () {
            return
        }
              let body = context.get_body_as_multipart().await;
           if let Ok(body) = body {
                let field = body.get_field("image");
            if let Some(field) = field {
                    let ref data = field.data;
                    let file_name = format!("./cache/fingerprint/{}.png",random_filename(12));
                    let file = tokio::fs::File::create(&file_name).await;
                    if let Ok(mut file) = file {
                        _= file.write_all(data).await.expect("can not write data to the specific file");
                       let start_time = SystemTime::now();
                        match crate::ai::match_using_sift_simple(
                                &file_name,
                               "mount/hres/hard") {
                                Ok((best,score))=>{
                                if let Some(best) = best {
                                   let id = get_file_id(&best);
                                let dif = format!("{:?}",SystemTime::now().duration_since(start_time));
                                  let res = serde_json::json!({
                                    "status":"success",
                                    "user-id":id,
                                    "matching-score":format!("{:?}",score),
                                    "matching-process-time":dif
                                });
                                 _= context.send_json(&res).await;
                                return
                                }
                            }
                                Err(_)=>{
                                let res = serde_json::json!({
                                    "status":"err",
                                    "msg":"invalid fingerprint please try again"
                                });
                                 _= context.send_json(&res).await;
                            }
                            }
                        return;
                    }
                    _=context.send_str("can not open file").await;
                }
            }

            _= context.send_str("not exist").await;
    }
    async fn ma(context)  {

          let body = context.get_body_as_multipart().await;
           if let Ok(body) = body {
                let field = body.get_field("image");
            if let Some(field) = field {
                    let ref data = field.data;
                    let file_name = format!("./cache/fingerprint/{}.png",random_filename(12));
                    let file = tokio::fs::File::create(&file_name).await;
                    if let Ok(mut file) = file {
                        _= file.write_all(data).await.expect("can not write data to the specific file");
                       let start_time = SystemTime::now();
                        match crate::ai::match_using_sift_simple(
                                &file_name,
                               "mount/hres/hard") {
                                Ok((best,score))=>{
                                if let Some(best) = best {
                                   let id = get_file_id(&best);
                                let dif = format!("{:?}",SystemTime::now().duration_since(start_time));

                                if score < 12 {
                                    let res = serde_json::json!({
                                        "status":"err",
                                        "msg":"the image is not even a fingerprint",
                                        "matching-time":dif
                                    });
                                    _= context.send_json(&res).await;
                                }

                                  let res = serde_json::json!({
                                    "status":"success",
                                    "user-id":id,
                                    "matching-score":format!("{:?}",score),
                                    "matching-time":dif
                                });
                                 _= context.send_json(&res).await;
                                return
                                }
                            }
                                Err(_)=>{
                                let res = serde_json::json!({
                                    "status":"err",
                                    "msg":"invalid fingerprint please try again"
                                });
                                 _= context.send_json(&res).await;
                            }
                            }
                        return;
                    }
                    _=context.send_str("can not open file").await;
                }
            }

            _= context.send_str("not exist").await;
    }
}