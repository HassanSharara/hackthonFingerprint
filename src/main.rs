pub mod ai;
mod ai_server;

#[tokio::main]
async fn main (){
  ai_server::run().await;
}

// fn main() -> opencv::Result<()> {
//  println!("🔍 Fingerprint Matcher - Final Fixed Version");
//
//  // Example usage (uncomment to test with actual images)
//  match
//
//  ai::match_using_sift_simple("mount/hres/easy/fingerprint_31_P005_fingerprint_10_easy_aug_09.png",
//                                "mount/hres/hard"
//  )
//
//  {
//   Ok((best_match, score)) => {
//    if let Some(filename) = best_match {
//     println!("✅ Best match: {} with score: {}", filename, score);
//    } else {
//     println!("❌ No match found");
//    }
//   }
//   Err(e) => println!("💥 Error: {}", e),
//  }
//
//  println!("✅ Program compiled successfully!");
//  Ok(())
// }